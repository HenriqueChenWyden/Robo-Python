import time
import pybullet as p
import pybullet_data
import pickle
import random
import numpy as np
import os
from sim.robot import DiffRobot
from sim.world import load_world
from mapping.occupancy import OccupancyGrid
from mapping.save_load import save_map, load_map
from logger.nodered import NodeRedLogger

RUN_ID = int(time.time())
USE_PREVIOUS_MAP = True

# ---------------------------------------------------------
# 1) Conectar PyBullet
# ---------------------------------------------------------
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane = load_world()

# ---------------------------------------------------------
# 2) Carregar o Robô Aspirador (Roomba)
# ---------------------------------------------------------
robot = DiffRobot(
    urdf=os.path.join(pybullet_data.getDataPath(), "base.urdf"),
    start_pos=(0, 0, 0.08),
    client_id=cid
)

# Dinâmica importante (senão o robô escorrega)
if len(robot.wheel_joints) >= 2:
    p.changeDynamics(robot.robot, robot.wheel_joints[0], lateralFriction=2.0)
    p.changeDynamics(robot.robot, robot.wheel_joints[1], lateralFriction=2.0)
p.changeDynamics(robot.robot, -1, lateralFriction=1.0)  # corpo

# ---------------------------------------------------------
# 3) Debug — listar todos os joints
# ---------------------------------------------------------
print("\n=== JOINTS DO ROBÔ ===")
for j in range(p.getNumJoints(robot.robot)):
    info = p.getJointInfo(robot.robot, j)
    print(
        "joint", j,
        "| name:", info[1],
        "| type:", info[2],
        "| limits:", (info[8], info[9])
    )
print("====================================\n")

# ---------------------------------------------------------
# 4) Carregar ou criar mapa
# ---------------------------------------------------------
try:
    if USE_PREVIOUS_MAP:
        grid, visits = load_map()
        occ = OccupancyGrid(existing_grid=grid, existing_visit=visits)
    else:
        raise FileNotFoundError
except:
    occ = OccupancyGrid()

# ---------------------------------------------------------
# 5) Logger -> NodeRED
# ---------------------------------------------------------
logger = NodeRedLogger(run_id=RUN_ID)

# ---------------------------------------------------------
# 6) Q-Learning
# ---------------------------------------------------------
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
    qdict = Q.get(state, {'forward':0,'left':0,'right':0})
    return max(qdict, key=qdict.get)

def update_q(state, action, reward, next_state):
    old = Q.get(state, {}).get(action, 0)
    future = max(Q.get(next_state, {'forward':0,'left':0,'right':0}).values())
    Q.setdefault(state, {})[action] = old + alpha * (reward + gamma * future - old)

def save_q_table():
    with open(Q_FILE, "wb") as f:
        pickle.dump(Q, f)

# ---------------------------------------------------------
# 7) Loop principal da simulação
# ---------------------------------------------------------
angles = np.linspace(-np.pi/2, np.pi/2, 9)  # 9 sensores frontais cobrindo 180°
dt = 1 / 30                                  # 30 FPS
max_dist = 1.0                               # metros

for step in range(5000):
    # Pose atual
    pose, orn = p.getBasePositionAndOrientation(robot.robot)

    # Leitura dos sensores (raycasts)
    readings = robot.range_readings(angles, max_dist=max_dist)

    # Atualiza mapa
    for a, dist in zip(angles, readings):
        occ.update(pose, a, dist)

    # Discretiza leituras para estado
    state = tuple(int(d*10) for d in readings)

    # Escolhe ação via Q-Learning
    action = choose_action(state)

    # --- Anti-colisão: se muito perto de parede, gira aleatoriamente ---
    min_dist = min(readings)
    if min_dist < 0.2:
        # Obstáculo muito próximo: girar aleatoriamente
        if random.random() < 0.5:
            lv, rv = -0.3, 0.3
        else:
            lv, rv = 0.3, -0.3
    else:
        # Converte ação do Q-learning em velocidades das rodas
        if action == 'forward':
            lv, rv = 10.5, 0.5
        elif action == 'left':
            lv, rv = -0.3, 0.3
        elif action == 'right':
            lv, rv = 0.3, -0.3

    # Move o robô
    robot.step(lv, rv, dt)

    # Nova leitura para próximo estado
    next_readings = robot.range_readings(angles, max_dist=max_dist)
    next_state = tuple(int(d*10) for d in next_readings)

    # Recompensa simples (evitar colisão)
    min_dist_next = min(next_readings)
    if min_dist_next < 0.1:
        reward = -1.0   # bateu
    elif min_dist_next < 0.3:
        reward = -0.5   # perto da parede
    else:
        reward = 0.1    # espaço livre

    # Atualiza Q-table
    update_q(state, action, reward, next_state)

    # Log para o Node-RED
    logger.send(pose, readings)

    # Avança simulação
    p.stepSimulation()
    time.sleep(dt)

# ---------------------------------------------------------
# 8) Finalização
# ---------------------------------------------------------
save_map(occ.grid, occ.visits)
save_q_table()
p.disconnect(cid)
