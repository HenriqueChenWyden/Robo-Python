import os
import time
import math
import pickle
import random
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pybullet as p


class DiffRobot:
    """Robô diferencial simples.

    - carrega URDF
    - identifica duas juntas de roda (revolute/continuous)
    - fornece step(left_vel, right_vel, duration) e range_readings(angles)

    left_vel/right_vel são interpretadas como velocidades lineares das rodas (m/s).
    """

    def __init__(
        self,
        urdf: str,
        start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.1),
        client_id: Optional[int] = None,
        wheel_radius: float = 0.05,
        max_force: float = 100.0,
    ) -> None:
        self.cid = client_id if client_id is not None else 0
        self.robot = p.loadURDF(urdf, start_pos, physicsClientId=self.cid)
        self.wheel_joints: List[int] = []
        self._find_wheels()
        self.use_base = len(self.wheel_joints) < 2
        self.wheel_base = 0.2  # distância aproximada entre rodas (m)
        self.wheel_radius = float(wheel_radius)
        self.max_force = float(max_force)

        # estado para suavização
        self._prev_left = 0.0
        self._prev_right = 0.0
        self._max_accel = 0.8  # m/s^2

        # tentar estimar wheel_base a partir da posição das juntas (se existirem)
        if len(self.wheel_joints) >= 2:
            try:
                p0 = p.getLinkState(self.robot, self.wheel_joints[0], physicsClientId=self.cid)[0]
                p1 = p.getLinkState(self.robot, self.wheel_joints[1], physicsClientId=self.cid)[0]
                dx = p0[0] - p1[0]
                dy = p0[1] - p1[1]
                est = math.hypot(dx, dy)
                if est > 1e-4:
                    self.wheel_base = est
            except Exception:
                pass

    def _find_wheels(self) -> None:
        n = p.getNumJoints(self.robot, physicsClientId=self.cid)
        for i in range(n):
            info = p.getJointInfo(self.robot, i, physicsClientId=self.cid)
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_CONTINUOUS:
                self.wheel_joints.append(i)
            if len(self.wheel_joints) >= 2:
                break

    def get_pose(self) -> Tuple[float, float, float]:
        pos, orn = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.cid)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        return (pos[0], pos[1], yaw)

    def step(self, left_vel: float, right_vel: float, duration: float) -> None:
        """Aplica comando por `duration` segundos.

        `left_vel` e `right_vel` em m/s (velocidades lineares das rodas).
        """
        dt = 1.0 / 240.0
        steps = max(1, int(duration / dt))

        def smooth(prev: float, target: float, max_accel: float, dt_: float) -> float:
            max_delta = max_accel * dt_
            delta = target - prev
            if abs(delta) > max_delta:
                return prev + math.copysign(max_delta, delta)
            return target

        if not self.use_base and len(self.wheel_joints) >= 2:
            curr_left = self._prev_left
            curr_right = self._prev_right
            for _ in range(steps):
                curr_left = smooth(curr_left, left_vel, self._max_accel, dt)
                curr_right = smooth(curr_right, right_vel, self._max_accel, dt)

                twl = curr_left / max(self.wheel_radius, 1e-6)
                twr = curr_right / max(self.wheel_radius, 1e-6)

                p.setJointMotorControl2(
                    self.robot,
                    self.wheel_joints[0],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=twl,
                    force=self.max_force,
                    physicsClientId=self.cid,
                )
                p.setJointMotorControl2(
                    self.robot,
                    self.wheel_joints[1],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=twr,
                    force=self.max_force,
                    physicsClientId=self.cid,
                )

                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)

            self._prev_left = curr_left
            self._prev_right = curr_right

        else:
            for _ in range(steps):
                self._prev_left = smooth(self._prev_left, left_vel, self._max_accel, dt)
                self._prev_right = smooth(self._prev_right, right_vel, self._max_accel, dt)

                lin = (self._prev_left + self._prev_right) / 2.0
                ang = (self._prev_right - self._prev_left) / max(self.wheel_base, 1e-6)

                p.resetBaseVelocity(
                    self.robot,
                    linearVelocity=[lin, 0.0, 0.0],
                    angularVelocity=[0.0, 0.0, ang],
                    physicsClientId=self.cid,
                )
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)

    def range_readings(
        self, angles: Iterable[float], max_dist: float = 5.0, height_offset: float = 0.1
    ) -> List[float]:
        px, py, yaw = self.get_pose()
        start_z = height_offset
        starts = []
        ends = []
        for a in angles:
            ang = yaw + a
            sx = px
            sy = py
            sz = start_z
            ex = sx + math.cos(ang) * max_dist
            ey = sy + math.sin(ang) * max_dist
            ez = sz
            starts.append((sx, sy, sz))
            ends.append((ex, ey, ez))

        results = p.rayTestBatch(starts, ends, physicsClientId=self.cid)
        dists: List[float] = []
        for res in results:
            hit_fraction = res[2]  # 1.0 == no hit
            if hit_fraction >= 0 and hit_fraction < 1.0:
                dists.append(hit_fraction * max_dist)
            else:
                dists.append(max_dist)
        return dists

    def attempt_climb(self, forward_speed: float = 0.35, duration: float = 1.2, extra_force: float = 300.0) -> None:
        """Tentativa simples de subir um obstáculo frontal.

        Estratégia:
        - aplica um impulso vertical curto no corpo (para 'subir' pequenas arestas)
        - aumenta temporariamente a força nos motores das rodas e aplica velocidade para frente
        - executa por `duration` segundos

        Observação: isso é heurístico para ambientes simples; ajuste `extra_force`
        e `forward_speed` conforme o URDF e o tipo de obstáculo.
        """
        dt = 1.0 / 240.0
        steps = max(1, int(duration / dt))

        # aplicar impulso vertical no início para ajudar a vencer pequenas bordas
        try:
            # força aplicada no centro de massa (linkIndex=-1)
            p.applyExternalForce(self.robot, -1, forceObj=[0.0, 0.0, 80.0], posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
        except Exception:
            pass

        if not self.use_base and len(self.wheel_joints) >= 2:
            # maior força temporária
            target_w = forward_speed / max(self.wheel_radius, 1e-6)
            torque = min(self.max_force + extra_force, 2000)
            for _ in range(steps):
                p.setJointMotorControl2(
                    self.robot,
                    self.wheel_joints[0],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_w,
                    force=torque,
                    physicsClientId=self.cid,
                )
                p.setJointMotorControl2(
                    self.robot,
                    self.wheel_joints[1],
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_w,
                    force=torque,
                    physicsClientId=self.cid,
                )
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)
        else:
            # fallback: avançar com base velocity e aplicar impulso adicional
            for _ in range(steps):
                p.resetBaseVelocity(self.robot, linearVelocity=[forward_speed, 0.0, 0.0], physicsClientId=self.cid)
                try:
                    p.applyExternalForce(self.robot, -1, forceObj=[0.0, 0.0, 40.0], posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.cid)
                except Exception:
                    pass
                p.stepSimulation(physicsClientId=self.cid)
                time.sleep(dt)


# --- Funções de Q-learning ---
Q_FILE = "q_table.pkl"
try:
    with open(Q_FILE, "rb") as f:
        Q = pickle.load(f)
except Exception:
    Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.1


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(["forward", "left", "right"])
    actions = Q.get(state, {"forward": 0, "left": 0, "right": 0})
    return max(actions, key=actions.get)


def update_q(state, action, reward, next_state):
    old = Q.get(state, {}).get(action, 0)
    future = max(Q.get(next_state, {"forward": 0, "left": 0, "right": 0}).values())
    Q.setdefault(state, {})[action] = old + alpha * (reward + gamma * future - old)


def save_q_table():
    with open(Q_FILE, "wb") as f:
        pickle.dump(Q, f)
