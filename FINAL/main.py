"""
differential_robot_with_wheels.py

Robô diferencial com rodas físicas (joints) em PyBullet, sensores ultrassônicos (ray tests),
controle PID simples (torque control nas juntas), algoritmo local de evasão e log de trajetória.

Requisitos:
    pip install pybullet numpy pandas
"""

import time
import math
import csv
import numpy as np
import pybullet as p
import pybullet_data

# -----------------------
# Parâmetros da simulação
# -----------------------
SIM_TIMESTEP = 1.0 / 240.0
SIM_REALTIME = True
SIM_DURATION = 11000.0  # segundos

# Robô / roda
WHEEL_RADIUS = 0.04  # m
WHEEL_BASE = 0.20    # m (distância entre rodas)
ROBOT_HALF_X = 0.15
ROBOT_HALF_Y = 0.12
ROBOT_HALF_Z = 0.05
MAX_WHEEL_TORQUE = 10.0  # N*m (limite do motor)

# Sensores ultrassom (ângulos relativos ao heading do robô)
# 5 sensores: esquerda, frente-esquerda, frente, frente-direita, direita
SENSOR_ANGLES = [
    math.radians(45),   # esquerda
    math.radians(22.5), # frente-esquerda
    0.0,                # frente
    math.radians(-22.5),# frente-direita
    math.radians(-45)   # direita
]

SENSOR_RANGE = 3.0  # m

# PID para rodas (gera torque a partir do erro de velocidade)
class PID:
    def __init__(self, kp, ki, kd, imax=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.imax = imax
        self.integral = 0.0
        self.prev = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev = 0.0

    def update(self, error, dt):
        self.integral += error * dt
        # anti-windup
        self.integral = max(min(self.integral, self.imax), -self.imax)
        deriv = (error - self.prev) / dt if dt > 0 else 0.0
        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        self.prev = error
        return out

def wrap_to_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

# -----------------------
# Classe do robô (com rodas físicas)
# -----------------------
class DifferentialRobot:
    def __init__(self, physics_client,
                 start_pos=(0,0,0.08),
                 start_yaw=0.0,
                 wheel_base=WHEEL_BASE,
                 wheel_radius=WHEEL_RADIUS,
                 sensor_angles=SENSOR_ANGLES,
                 sensor_range=SENSOR_RANGE):
        self.p = physics_client
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.sensor_angles = sensor_angles
        self.sensor_range = sensor_range

        # PID controllers for each wheel (target: angular velocity [rad/s])
        self.left_pid = PID(kp=1.8, ki=0.0, kd=0.03, imax=1.0)
        self.right_pid = PID(kp=1.8, ki=0.0, kd=0.03, imax=1.0)

        # targets
        self.target_left = 0.0
        self.target_right = 0.0

        # build multibody: base + 2 wheel links
        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ROBOT_HALF_X, ROBOT_HALF_Y, ROBOT_HALF_Z])
        body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[ROBOT_HALF_X, ROBOT_HALF_Y, ROBOT_HALF_Z],
                                          rgbaColor=[0.2,0.6,0.9,1.0])

        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.wheel_radius, height=0.04)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.wheel_radius, length=0.04,
                                        rgbaColor=[0.05, 0.05, 0.05, 1.0])

        # linkPositions are relative to base frame
        link_positions = [
            [0.0,  self.wheel_base/2.0, -ROBOT_HALF_Z],   # left wheel (slightly lower)
            [0.0, -self.wheel_base/2.0, -ROBOT_HALF_Z]    # right wheel
        ]
        # orientation: rotate cylinder so axis aligns with joint axis (y-axis)
        link_orientations = [
            p.getQuaternionFromEuler([math.pi/2.0, 0, 0]),
            p.getQuaternionFromEuler([math.pi/2.0, 0, 0])
        ]

        self.body = p.createMultiBody(
            baseMass=3.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=[start_pos[0], start_pos[1], start_pos[2]],
            baseOrientation=p.getQuaternionFromEuler([0, 0, start_yaw]),
            linkMasses=[0.3, 0.3],
            linkCollisionShapeIndices=[wheel_col, wheel_col],
            linkVisualShapeIndices=[wheel_vis, wheel_vis],
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0,0,0],[0,0,0]],
            linkInertialFrameOrientations=[p.getQuaternionFromEuler([0,0,0]), p.getQuaternionFromEuler([0,0,0])],
            linkParentIndices=[0, 0],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0,1,0], [0,1,0]]
        )

        # wheel joint indices are 0 and 1 (links order)
        self.left_wheel_idx = 0
        self.right_wheel_idx = 1

        # ensure wheel collisions don't produce unwanted friction artifacts
        p.changeDynamics(self.body, self.left_wheel_idx, lateralFriction=0.9, rollingFriction=0.0001, spinningFriction=0.0)
        p.changeDynamics(self.body, self.right_wheel_idx, lateralFriction=0.9, rollingFriction=0.0001, spinningFriction=0.0)

        # disable default velocity motors - we'll apply torque control
        p.setJointMotorControl2(self.body, self.left_wheel_idx, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.body, self.right_wheel_idx, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # internal state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body)
        bx, by, bz = base_pos
        _, _, byaw = p.getEulerFromQuaternion(base_orn)

        self.x = bx
        self.y = by
        self.yaw = byaw

        # odometry estimate
        self.odom_x = self.x
        self.odom_y = self.y
        self.odom_yaw = self.yaw

        # wheel encoder noise / odom noise
        self.encoder_noise_std = 0.005  # rad/s
        self.odom_noise_trans = 0.001
        self.odom_noise_rot = math.radians(0.3)

    def set_wheel_targets(self, left_rad_s, right_rad_s):
        self.target_left = left_rad_s
        self.target_right = right_rad_s

    def apply_pid_and_torque(self, dt):
        # read actual wheel angular velocities from joint states
        lj = p.getJointState(self.body, self.left_wheel_idx)
        rj = p.getJointState(self.body, self.right_wheel_idx)
        vel_left = lj[1] + np.random.normal(0, self.encoder_noise_std)
        vel_right = rj[1] + np.random.normal(0, self.encoder_noise_std)

        # compute errors
        err_l = self.target_left - vel_left
        err_r = self.target_right - vel_right

        # pid -> torque
        torque_l = self.left_pid.update(err_l, dt)
        torque_r = self.right_pid.update(err_r, dt)

        # saturate torque to motor limits
        torque_l = max(min(torque_l, MAX_WHEEL_TORQUE), -MAX_WHEEL_TORQUE)
        torque_r = max(min(torque_r, MAX_WHEEL_TORQUE), -MAX_WHEEL_TORQUE)

        # apply torques (TORQUE_CONTROL)
        p.setJointMotorControl2(self.body, self.left_wheel_idx,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque_l)
        p.setJointMotorControl2(self.body, self.right_wheel_idx,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque_r)

    def update_internal_pose_from_sim(self):
        # update pose from pybullet base
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body)
        bx, by, bz = base_pos
        _, _, byaw = p.getEulerFromQuaternion(base_orn)
        self.x = bx
        self.y = by
        self.yaw = wrap_to_pi(byaw)

    def update_odometry_from_encoders(self, dt):
        # read wheel angular velocities
        lj = p.getJointState(self.body, self.left_wheel_idx)
        rj = p.getJointState(self.body, self.right_wheel_idx)
        wl = lj[1] + np.random.normal(0, self.encoder_noise_std)
        wr = rj[1] + np.random.normal(0, self.encoder_noise_std)
        dl = self.wheel_radius * wl * dt + np.random.normal(0, self.odom_noise_trans)
        dr = self.wheel_radius * wr * dt + np.random.normal(0, self.odom_noise_trans)
        dc = (dr + dl) / 2.0
        dyaw = (dr - dl) / self.wheel_base + np.random.normal(0, self.odom_noise_rot)
        self.odom_x += dc * math.cos(self.odom_yaw)
        self.odom_y += dc * math.sin(self.odom_yaw)
        self.odom_yaw = wrap_to_pi(self.odom_yaw + dyaw)

    def read_ultrasonic_sensors(self):
        results = []
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body)
        bx, by, bz = base_pos
        _, _, yaw = p.getEulerFromQuaternion(base_orn)
        from_positions = []
        to_positions = []
        # small height to avoid ground hits
        sz = bz + 0.02
        for ang in self.sensor_angles:
            ang_world = yaw + ang
            sx = bx
            sy = by
            tx = sx + self.sensor_range * math.cos(ang_world)
            ty = sy + self.sensor_range * math.sin(ang_world)
            tz = sz
            from_positions.append([sx, sy, sz])
            to_positions.append([tx, ty, tz])
        ray_results = p.rayTestBatch(from_positions, to_positions)
        for rr in ray_results:
            hit_fraction = rr[2]
            if hit_fraction < 1.0:
                dist = hit_fraction * self.sensor_range
            else:
                dist = float('inf')
            results.append(dist)
        return results

    def read_pose(self):
        # true pose and odom pose
        return (self.x, self.y, self.yaw), (self.odom_x, self.odom_y, self.odom_yaw)

    def read_velocity(self):
        # approximate linear and angular velocity from wheel joint velocities
        lj = p.getJointState(self.body, self.left_wheel_idx)
        rj = p.getJointState(self.body, self.right_wheel_idx)
        wl = lj[1]
        wr = rj[1]
        v_lin = self.wheel_radius * 0.5 * (wl + wr)
        v_ang = self.wheel_radius * (wr - wl) / self.wheel_base
        return v_lin, v_ang

# -----------------------
# Ambiente (planta da casa)
# -----------------------
def build_environment():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    walls = []
    wall_h = 0.4
    wt = 0.05

    def add_wall(center, half_extents):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8,0.8,0.8,1])
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=center)
        return body

    # room ~4m x 3m
    walls.append(add_wall([0.0, 1.5, wall_h/2], [2.0, wt, wall_h/2]))
    walls.append(add_wall([0.0, -1.5, wall_h/2], [2.0, wt, wall_h/2]))
    walls.append(add_wall([1.75, 0.0, wall_h/2], [wt, 1.5, wall_h/2]))
    walls.append(add_wall([-1.75, 0.0, wall_h/2], [wt, 1.5, wall_h/2]))

    # obstacles (furniture)
    obstacles = []
    obstacles.append(add_wall([0.5, 0.5, 0.2], [0.3, 0.3, 0.2]))
    obstacles.append(add_wall([-0.7, -0.3, 0.15], [0.15, 0.15, 0.15]))
    obstacles.append(add_wall([-0.2, 0.9, 0.2], [0.2, 0.4, 0.2]))
    return plane, walls, obstacles

# -----------------------
# Controle local (evasão simples)
# -----------------------
class LocalController:
    def __init__(self, robot: DifferentialRobot):
        self.robot = robot
        self.cruise_speed = 0.20  # m/s
        self.sensor_threshold = 0.45  # m
        self.avoid_angular_speed = 1.0  # rad/s (used to set wheel targets relatively)
        # convenience: convert to wheel angular speeds when needed
        self.max_wheel_rad = 20.0  # rad/s safe cap

    def compute_wheel_targets(self):
        dists = self.robot.read_ultrasonic_sensors()
        # if any obstacle close -> avoid
        close_idxs = [i for i, d in enumerate(dists) if d < self.sensor_threshold]
        if close_idxs:
            # take sensor with minimum reading
            min_idx = int(np.argmin(np.array(dists)))
            angle = self.robot.sensor_angles[min_idx]
            # if obstacle on left (negative angle) -> turn right
            if angle < 0:
                # right spin: left wheel forward, right wheel backward
                vl = self.avoid_angular_speed * (self.robot.wheel_base/2.0)  # linear m/s
                vr = -vl
            else:
                vl = -self.avoid_angular_speed * (self.robot.wheel_base/2.0)
                vr = -vl
            # convert linear wheel tangential speed to wheel angular speed (rad/s)
            left_rad = vl / self.robot.wheel_radius
            right_rad = vr / self.robot.wheel_radius
            left_rad = max(min(left_rad, self.max_wheel_rad), -self.max_wheel_rad)
            right_rad = max(min(right_rad, self.max_wheel_rad), -self.max_wheel_rad)
            self.robot.set_wheel_targets(left_rad, right_rad)
        else:
            # move forward
            wheel_rad = self.cruise_speed / self.robot.wheel_radius
            wheel_rad = max(min(wheel_rad, self.max_wheel_rad), -self.max_wheel_rad)
            self.robot.set_wheel_targets(wheel_rad, wheel_rad)

# -----------------------
# Loop principal
# -----------------------
def run_simulation():
    physicsClient = p.connect(p.GUI)
    p.setGravity(0,0,-9.81)
    p.setTimeStep(SIM_TIMESTEP)
    p.setRealTimeSimulation(0)

    build_environment()

    # create robot
    robot = DifferentialRobot(physicsClient, start_pos=(-1.0, -0.6, 0.08), start_yaw=math.radians(30))

    controller = LocalController(robot)

    # CSV log
    csv_filename = "trajectory_log_wheels.csv"
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["time", "x", "y", "yaw", "odom_x", "odom_y", "odom_yaw", "v_lin", "v_ang"]
    for i in range(len(robot.sensor_angles)):
        header.append(f"sensor_{i}")
    csv_writer.writerow(header)

    sim_time = 0.0

    while sim_time < SIM_DURATION:
        t0 = time.time()
        # compute control (target wheel angular velocities)
        controller.compute_wheel_targets()
        # apply pid -> torque to joints
        robot.apply_pid_and_torque(SIM_TIMESTEP)
        # step simulation
        p.stepSimulation()
        # update pose (read from sim)
        robot.update_internal_pose_from_sim()
        # update odometry from encoders
        robot.update_odometry_from_encoders(SIM_TIMESTEP)

        # read sensors & velocities
        sensors = robot.read_ultrasonic_sensors()
        (x,y,yaw), (ox,oy,oyaw) = robot.read_pose()
        v_lin, v_ang = robot.read_velocity()

        row = [sim_time, x, y, yaw, ox, oy, oyaw, v_lin, v_ang] + sensors
        csv_writer.writerow(row)

        sim_time += SIM_TIMESTEP

        if SIM_REALTIME:
            elapsed = time.time() - t0
            sleep_t = max(0.0, SIM_TIMESTEP - elapsed)
            time.sleep(sleep_t)

    csv_file.close()
    print(f"Simulação completa. Log salvo em: {csv_filename}")
    print("Feche a janela do PyBullet para encerrar.")
    while p.isConnected():
        time.sleep(0.1)
    p.disconnect()

if __name__ == "__main__":
    run_simulation()