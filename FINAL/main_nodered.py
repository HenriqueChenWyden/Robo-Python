import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import json
import paho.mqtt.client as mqtt

# -----------------------
# Parâmetros da simulação
# -----------------------
SIM_TIMESTEP = 1.0 / 100.0
SIM_REALTIME = True
SIM_DURATION = 75.0  # segundos

# Robô / roda
WHEEL_RADIUS = 0.04  # m
WHEEL_BASE = 0.20    # m
ROBOT_HALF_X = 0.15
ROBOT_HALF_Y = 0.12
ROBOT_HALF_Z = 0.05
MAX_WHEEL_TORQUE = 10.0  # N*m

# Sensores ultrassom (ângulos relativos ao heading do robô)
SENSOR_ANGLES = [
    math.radians(45), math.radians(22.5), 0.0,
    math.radians(-22.5), math.radians(-45)
]
SENSOR_RANGE = 3.0  # m

# -----------------------
# PID para rodas
# -----------------------
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
        self.integral = max(min(self.integral, self.imax), -self.imax)
        deriv = (error - self.prev) / dt if dt>0 else 0.0
        out = self.kp*error + self.ki*self.integral + self.kd*deriv
        self.prev = error
        return out

def wrap_to_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

# -----------------------
# Classe do robô
# -----------------------
class DifferentialRobot:
    def __init__(self, physics_client, start_pos=(0,0,ROBOT_HALF_Z), start_yaw=0.0):
        self.p = physics_client
        self.wheel_base = WHEEL_BASE
        self.wheel_radius = WHEEL_RADIUS
        self.sensor_angles = SENSOR_ANGLES
        self.sensor_range = SENSOR_RANGE

        self.left_pid = PID(1.8,0.0,0.03,1.0)
        self.right_pid = PID(1.8,0.0,0.03,1.0)
        self.target_left = 0.0
        self.target_right = 0.0

        body_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[ROBOT_HALF_X,ROBOT_HALF_Y,ROBOT_HALF_Z])
        body_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[ROBOT_HALF_X,ROBOT_HALF_Y,ROBOT_HALF_Z], rgbaColor=[0.2,0.6,0.9,1.0])
        wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.wheel_radius, height=0.04)
        wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=self.wheel_radius, length=0.04, rgbaColor=[0.05,0.05,0.05,1.0])

        link_positions = [[0.0, self.wheel_base/2.0, -ROBOT_HALF_Z],
                          [0.0, -self.wheel_base/2.0, -ROBOT_HALF_Z]]
        link_orientations = [p.getQuaternionFromEuler([math.pi/2,0,0]), p.getQuaternionFromEuler([math.pi/2,0,0])]

        self.body = p.createMultiBody(
            baseMass=3.0, baseCollisionShapeIndex=body_collision, baseVisualShapeIndex=body_visual,
            basePosition=start_pos, baseOrientation=p.getQuaternionFromEuler([0,0,start_yaw]),
            linkMasses=[0.3,0.3], linkCollisionShapeIndices=[wheel_col,wheel_col],
            linkVisualShapeIndices=[wheel_vis,wheel_vis], linkPositions=link_positions,
            linkOrientations=link_orientations, linkInertialFramePositions=[[0,0,0],[0,0,0]],
            linkInertialFrameOrientations=[p.getQuaternionFromEuler([0,0,0]),p.getQuaternionFromEuler([0,0,0])],
            linkParentIndices=[0,0], linkJointTypes=[p.JOINT_REVOLUTE,p.JOINT_REVOLUTE],
            linkJointAxis=[[0,1,0],[0,1,0]]
        )

        self.left_wheel_idx=0
        self.right_wheel_idx=1
        p.setJointMotorControl2(self.body,self.left_wheel_idx,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
        p.setJointMotorControl2(self.body,self.right_wheel_idx,p.VELOCITY_CONTROL,targetVelocity=0,force=0)

        self.x, self.y, self.yaw = start_pos[0], start_pos[1], start_yaw
        self.odom_x, self.odom_y, self.odom_yaw = self.x, self.y, self.yaw
        self.encoder_noise_std = 0.005
        self.odom_noise_trans = 0.001
        self.odom_noise_rot = math.radians(0.3)

    def set_wheel_targets(self,left_rad_s,right_rad_s):
        self.target_left = left_rad_s
        self.target_right = right_rad_s

    def apply_pid_and_torque(self, dt):
        lj = p.getJointState(self.body,self.left_wheel_idx)
        rj = p.getJointState(self.body,self.right_wheel_idx)
        vel_left = lj[1] + np.random.normal(0,self.encoder_noise_std)
        vel_right = rj[1] + np.random.normal(0,self.encoder_noise_std)
        err_l = self.target_left - vel_left
        err_r = self.target_right - vel_right
        torque_l = max(min(self.left_pid.update(err_l, dt), MAX_WHEEL_TORQUE), -MAX_WHEEL_TORQUE)
        torque_r = max(min(self.right_pid.update(err_r, dt), MAX_WHEEL_TORQUE), -MAX_WHEEL_TORQUE)
        p.setJointMotorControl2(self.body,self.left_wheel_idx,p.TORQUE_CONTROL,force=torque_l)
        p.setJointMotorControl2(self.body,self.right_wheel_idx,p.TORQUE_CONTROL,force=torque_r)

    def update_internal_pose_from_sim(self):
        base_pos,_ = p.getBasePositionAndOrientation(self.body)
        self.x, self.y = base_pos[0], base_pos[1]

    def update_odometry_from_encoders(self, dt):
        lj = p.getJointState(self.body,self.left_wheel_idx)
        rj = p.getJointState(self.body,self.right_wheel_idx)
        wl = lj[1] + np.random.normal(0,self.encoder_noise_std)
        wr = rj[1] + np.random.normal(0,self.encoder_noise_std)
        dl = self.wheel_radius*wl*dt + np.random.normal(0,self.odom_noise_trans)
        dr = self.wheel_radius*wr*dt + np.random.normal(0,self.odom_noise_trans)
        dc = (dl+dr)/2
        dyaw = (dr-dl)/self.wheel_base + np.random.normal(0,self.odom_noise_rot)
        self.odom_x += dc*math.cos(self.odom_yaw)
        self.odom_y += dc*math.sin(self.odom_yaw)
        self.odom_yaw = wrap_to_pi(self.odom_yaw+dyaw)

    def read_ultrasonic_sensors(self):
        results=[]
        base_pos, base_orn = p.getBasePositionAndOrientation(self.body)
        bx,by,bz=base_pos
        _,_,yaw=p.getEulerFromQuaternion(base_orn)
        sz = bz+0.02
        from_positions=[]
        to_positions=[]
        for ang in self.sensor_angles:
            ang_world = yaw + ang
            tx = bx + self.sensor_range*math.cos(ang_world)
            ty = by + self.sensor_range*math.sin(ang_world)
            tz = sz
            from_positions.append([bx,by,sz])
            to_positions.append([tx,ty,tz])
        ray_results = p.rayTestBatch(from_positions,to_positions)
        for rr in ray_results:
            hit_fraction = rr[2]
            dist = hit_fraction*self.sensor_range if hit_fraction<1.0 else float('inf')
            results.append(dist)
        return results

    def read_velocity(self):
        lj = p.getJointState(self.body,self.left_wheel_idx)
        rj = p.getJointState(self.body,self.right_wheel_idx)
        wl,wr = lj[1], rj[1]
        v_lin = self.wheel_radius*0.5*(wl+wr)
        v_ang = self.wheel_radius*(wr-wl)/self.wheel_base
        return v_lin,v_ang

# -----------------------
# Mapa de ocupação
# -----------------------
class CoverageMap:
    def __init__(self, x_range=(-2,2), y_range=(-1.5,1.5), cell_size=0.05):
        self.cell_size = cell_size
        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range
        self.nx = int((self.xmax-self.xmin)/cell_size)
        self.ny = int((self.ymax-self.ymin)/cell_size)
        self.map = np.zeros((self.nx,self.ny), dtype=float)

    def pos_to_idx(self, x, y):
        ix = int((x-self.xmin)/self.cell_size)
        iy = int((y-self.ymin)/self.cell_size)
        return ix, iy

    def update(self, x, y, dt=1.0):
        ix, iy = self.pos_to_idx(x, y)
        if 0<=ix<self.nx and 0<=iy<self.ny:
            self.map[ix, iy] += dt

    def coverage_ratio(self):
        return np.sum(self.map>0)/self.map.size

    def get_least_visited_direction(self, x, y):
        ix, iy = self.pos_to_idx(x, y)
        neighborhood = self.map[max(0,ix-1):min(self.nx,ix+2), max(0,iy-1):min(self.ny,iy+2)]
        min_idx = np.unravel_index(np.argmin(neighborhood), neighborhood.shape)
        dx = (min_idx[0] + max(0,ix-1) - ix) * self.cell_size
        dy = (min_idx[1] + max(0,iy-1) - iy) * self.cell_size
        return dx, dy

# -----------------------
# Controle local
# -----------------------
class LocalController:
    def __init__(self, robot: DifferentialRobot, coverage_map:CoverageMap=None):
        self.robot = robot
        self.cruise_speed = 0.2
        self.sensor_threshold = 0.45
        self.avoid_angular_speed = 1.0
        self.max_wheel_rad = 20.0
        self.coverage_map = coverage_map

    def compute_wheel_targets(self):
        dists = self.robot.read_ultrasonic_sensors()
        close_idxs = [i for i,d in enumerate(dists) if d<self.sensor_threshold]
        if close_idxs:
            min_idx = int(np.argmin(np.array(dists)))
            angle = self.robot.sensor_angles[min_idx]
            if angle<0:
                vl = self.avoid_angular_speed*(self.robot.wheel_base/2.0)
                vr = -vl
            else:
                vl = -self.avoid_angular_speed*(self.robot.wheel_base/2.0)
                vr = vl
        else:
            if self.coverage_map:
                dx, dy = self.coverage_map.get_least_visited_direction(self.robot.x, self.robot.y)
                target_angle = math.atan2(dy, dx)
                angle_diff = wrap_to_pi(target_angle - self.robot.yaw)
                vl = self.cruise_speed - angle_diff
                vr = self.cruise_speed + angle_diff
            else:
                vl = vr = self.cruise_speed

        left_rad = max(min(vl/self.robot.wheel_radius,self.max_wheel_rad),-self.max_wheel_rad)
        right_rad = max(min(vr/self.robot.wheel_radius,self.max_wheel_rad),-self.max_wheel_rad)
        self.robot.set_wheel_targets(left_rad, right_rad)

# -----------------------
# Ambiente
# -----------------------
def build_environment():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    wall_h=0.4
    wt=0.05
    def add_wall(center,half_extents):
        col = p.createCollisionShape(p.GEOM_BOX,halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX,halfExtents=half_extents,rgbaColor=[0.8,0.8,0.8,1])
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=center)
    add_wall([0.0,1.5,wall_h/2],[2.0,wt,wall_h/2])
    add_wall([0.0,-1.5,wall_h/2],[2.0,wt,wall_h/2])
    add_wall([1.75,0.0,wall_h/2],[wt,1.5,wall_h/2])
    add_wall([-1.75,0.0,wall_h/2],[wt,1.5,wall_h/2])
    add_wall([0.5,0.5,0.2],[0.3,0.3,0.2])
    add_wall([-0.7,-0.3,0.15],[0.15,0.15,0.15])
    add_wall([-0.2,0.9,0.2],[0.2,0.4,0.2])

# -----------------------
# Loop principal
# -----------------------
def run_simulation_node_red(executions=3):
    physicsClient = p.connect(p.GUI)
    p.setGravity(0,0,-9.81)
    p.setTimeStep(SIM_TIMESTEP)
    p.setRealTimeSimulation(0)

    build_environment()
    coverage_map = CoverageMap()

    # MQTT
    mqtt_client = mqtt.Client()
    mqtt_client.connect("localhost", 1883, 60)

    for run_idx in range(executions):
        print(f"Execução {run_idx+1}/{executions}")
        robot = DifferentialRobot(physicsClient, start_pos=(-1.0,-0.6,ROBOT_HALF_Z), start_yaw=math.radians(30))
        controller = LocalController(robot, coverage_map if run_idx>0 else None)

        sim_time=0.0
        while sim_time<SIM_DURATION:
            t0=time.time()
            controller.compute_wheel_targets()
            robot.apply_pid_and_torque(SIM_TIMESTEP)
            p.stepSimulation()
            robot.update_internal_pose_from_sim()
            robot.update_odometry_from_encoders(SIM_TIMESTEP)
            coverage_map.update(robot.x, robot.y, dt=SIM_TIMESTEP)

            # Envio MQTT
            msg = {
                "time": sim_time,
                "x": robot.x,
                "y": robot.y,
                "yaw": robot.yaw,
                "odom_x": robot.odom_x,
                "odom_y": robot.odom_y,
                "odom_yaw": robot.odom_yaw,
                "v_lin": robot.read_velocity()[0],
                "v_ang": robot.read_velocity()[1],
                "coverage_ratio": coverage_map.coverage_ratio(),
                "cell_time": coverage_map.map.tolist()
            }
            mqtt_client.publish("robot/simulation", json.dumps(msg))

            sim_time+=SIM_TIMESTEP
            if SIM_REALTIME:
                elapsed=time.time()-t0
                time.sleep(max(0.0,SIM_TIMESTEP-elapsed))

        print(f"Cobertura após execução {run_idx+1}: {coverage_map.coverage_ratio()*100:.2f}%")

    p.disconnect()
    mqtt_client.disconnect()

if __name__=="__main__":
    run_simulation_node_red()