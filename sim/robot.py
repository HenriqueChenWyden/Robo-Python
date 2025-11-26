import os
import time
import math
import pickle
import random
import numpy as np
import pybullet as p

class DiffRobot:
    """
    Robô diferencial simples:
    - carrega URDF
    - identifica duas juntas de roda (revolute/continuous)
    - fornece step(left_vel, right_vel, duration) e range_readings(angles)
    """
    def __init__(self, urdf, start_pos=(0,0,0.1), client_id=None):
        self.cid = client_id if client_id is not None else 0
        self.robot = p.loadURDF(urdf, start_pos, physicsClientId=self.cid)
        self.wheel_joints = []
        self._find_wheels()
        self.use_base = len(self.wheel_joints) < 2
        self.wheel_base = 0.2  # distância aproximada entre rodas

    def _find_wheels(self):
        n = p.getNumJoints(self.robot, physicsClientId=self.cid)
        for i in range(n):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE:
                self.wheel_joints.append(i)
            if len(self.wheel_joints) >= 2:
                break

    def get_pose(self):
            pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
            euler = p.getEulerFromQuaternion(orn)
            yaw = euler[2]
            return (pos[0], pos[1], yaw)

    def step(self, left_vel, right_vel, duration):
        dt = 1.0 / 240.0
        steps = max(1, int(duration / dt))
        if not self.use_base:
            # controle por juntas
            p.setJointMotorControl2(self.robot, self.wheel_joints[0],
                                    controlMode=p.VELOCITY_CONTROL, targetVelocity=left_vel,
                                    force=100, physicsClientId=self.cid)
            p.setJointMotorControl2(self.robot, self.wheel_joints[1],
                                    controlMode=p.VELOCITY_CONTROL, targetVelocity=right_vel,
                                    force=100, physicsClientId=self.cid)
            for _ in range(steps):
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)
        else:
            # fallback: ajustar base
            lin = (left_vel + right_vel) / 2.0
            ang = (right_vel - left_vel) / max(self.wheel_base, 1e-6)
            for _ in range(steps):
                p.resetBaseVelocity(self.robot, linearVelocity=[lin,0,0],
                                    angularVelocity=[0,0,ang], physicsClientId=self.cid)
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)

    def range_readings(self, angles, max_dist=5.0, height_offset=0.1):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        yaw = p.getEulerFromQuaternion(base_orn)[2]

        from_positions = []
        to_positions = []
        for a in angles:
            ang = yaw + a
            from_p = [base_pos[0], base_pos[1], base_pos[2] + height_offset]
            to_p = [from_p[0] + max_dist * math.cos(ang),
                    from_p[1] + max_dist * math.sin(ang),
                    from_p[2]]
            from_positions.append(from_p)
            to_positions.append(to_p)

        results = p.rayTestBatch(from_positions, to_positions, physicsClientId=self.cid)
        dists = []
        for i, r in enumerate(results):
            hit_id = r[0]
            hit_pos = r[3]
            if hit_id == -1:
                dists.append(max_dist)
            else:
                fx = from_positions[i]
                dist = math.sqrt((hit_pos[0]-fx[0])**2 + (hit_pos[1]-fx[1])**2 + (hit_pos[2]-fx[2])**2)
                dists.append(dist)
        return dists

# --- Funções de Q-learning --- 
Q_FILE = "q_table.pkl"
try:
    with open(Q_FILE, "rb") as f:
        Q = pickle.load(f)
except:
    Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.1

def choose_action(state):
    if random.random() < epsilon:
        return random.choice(['forward','left','right'])
    return max(Q.get(state, {'forward':0,'left':0,'right':0}), key=Q.get(state, {'forward':0,'left':0,'right':0}).get)

def update_q(state, action, reward, next_state):
    old = Q.get(state, {}).get(action, 0)
    future = max(Q.get(next_state, {'forward':0,'left':0,'right':0}).values())
    Q.setdefault(state, {})[action] = old + alpha * (reward + gamma * future - old)

def save_q_table():
    with open(Q_FILE, "wb") as f:
        pickle.dump(Q, f)
