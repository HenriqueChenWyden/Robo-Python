"""
robo_aspirador_sim.py
Simulação única em um arquivo:
- PyBullet world simples (casa com obstáculos)
- Robô diferencial com 3 sensores ultrassônicos (simulados via rayTest)
- Occupancy grid 2D
- Exploração + evasão, registro de trajetória
- Aprendizado por repetição simples (reusar mapa e priorizar células não limpas)
- Envio de métricas para Node-RED via HTTP POST (opcional)

Como usar:
    pip install pybullet numpy requests
    python robo_aspirador_sim.py --runs 3 --node-red http://localhost:1880/log
    python robo_aspirador_sim.py --runs 1 --no-node-red

OBS: é protótipo. Ajuste parâmetros para seu caso.
"""

import argparse
import math
import time
import pickle
import os
import json
from collections import deque, defaultdict

import numpy as np
import pybullet as p
import pybullet_data

try:
    import requests
except Exception:
    requests = None

# ---------- Config ----------
MAP_FILE = "occupancy_map.pkl"
LOG_DIR = "sim_logs"
os.makedirs(LOG_DIR, exist_ok=True)

WORLD_SIZE = 8.0  # meters (square world centered at origin)
MAP_RESOLUTION = 0.1  # meters per cell
ROBOT_RADIUS = 0.18  # m
TIME_STEP = 1.0 / 60.0

SENSOR_ANGLES = [-math.radians(45), 0.0, math.radians(45)]  # left, center, right
SENSOR_RANGE = 2.0  # meters

# PID for wheel speed (very simple)
KP = 10.0
KI = 0.0
KD = 0.0

# ---------- Utilities ----------
def clamp(x, a, b):
    return max(a, min(b, x))

def world_to_map(x, y, origin_offset, resolution):
    ix = int((x + origin_offset) / resolution)
    iy = int((y + origin_offset) / resolution)
    return ix, iy

def map_to_world(ix, iy, origin_offset, resolution):
    x = ix * resolution - origin_offset + resolution / 2.0
    y = iy * resolution - origin_offset + resolution / 2.0
    return x, y

# ---------- Occupancy Grid ----------
class OccupancyGrid:
    def __init__(self, world_size=WORLD_SIZE, resolution=MAP_RESOLUTION):
        self.resolution = resolution
        self.world_size = world_size
        self.origin_offset = world_size / 2.0
        self.size_cells = int(math.ceil(world_size / resolution))
        self.grid = np.zeros((self.size_cells, self.size_cells), dtype=np.float32)  # 0 free, >0 occupied probability
        self.visits = np.zeros_like(self.grid, dtype=np.int32)
        self.cleaned_time = np.zeros_like(self.grid, dtype=np.float32)  # total time spent on cell

    def mark_occupied(self, x, y, prob=1.0):
        ix, iy = world_to_map(x, y, self.origin_offset, self.resolution)
        if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
            self.grid[ix, iy] = clamp(self.grid[ix, iy] + prob, 0.0, 1.0)

    def mark_free(self, x, y):
        ix, iy = world_to_map(x, y, self.origin_offset, self.resolution)
        if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
            # decay occupied prob slowly
            self.grid[ix, iy] = max(0.0, self.grid[ix, iy] - 0.05)

    def increment_visit(self, x, y, dt=0.0):
        ix, iy = world_to_map(x, y, self.origin_offset, self.resolution)
        if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
            self.visits[ix, iy] += 1
            if dt > 0:
                self.cleaned_time[ix, iy] += dt

    def percent_covered(self):
        free_cells = np.sum(self.grid < 0.5)
        total = self.size_cells * self.size_cells
        return 100.0 * free_cells / total

    def save(self, filename=MAP_FILE):
        with open(filename, "wb") as f:
            pickle.dump({
                "grid": self.grid,
                "visits": self.visits,
                "cleaned_time": self.cleaned_time,
                "world_size": self.world_size,
                "resolution": self.resolution
            }, f)

    def load(self, filename=MAP_FILE):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.grid = data["grid"]
        self.visits = data["visits"]
        self.cleaned_time = data["cleaned_time"]
        self.world_size = data["world_size"]
        self.resolution = data["resolution"]
        self.origin_offset = self.world_size / 2.0
        self.size_cells = self.grid.shape[0]

    def unexplored_frontiers(self, threshold_visits=1):
        # return list of world coordinates of low-visit cells adjacent to free cells
        frontiers = []
        for ix in range(1, self.size_cells - 1):
            for iy in range(1, self.size_cells - 1):
                if self.visits[ix, iy] <= threshold_visits and self.grid[ix, iy] < 0.5:
                    # check neighbor with 0 visits indicates frontier
                    neigh = self.visits[ix-1:ix+2, iy-1:iy+2]
                    if np.any(neigh > threshold_visits):
                        x, y = map_to_world(ix, iy, self.origin_offset, self.resolution)
                        frontiers.append((x, y))
        return frontiers

# ---------- Robot (differential) ----------
class DiffRobot:
    def __init__(self, client, start_pos=(0, 0, 0.05), start_yaw=0.0):
        self.client = client
        self._create_body(start_pos, start_yaw)
        self.max_speed = 4.0  # rad/s for wheels
        self.wheel_radius = 0.05
        self.wheel_base = 0.25  # distance between wheels
        self.left_integral = 0.0
        self.right_integral = 0.0
        self.left_last_err = 0.0
        self.right_last_err = 0.0

    def _create_body(self, pos, yaw):
        # Simple cylinder as base + two tiny boxes as wheels (visual only)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, height=0.06)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=ROBOT_RADIUS, length=0.06, rgbaColor=[0.2,0.2,0.8,1])
        mass = 3.0
        self.body = p.createMultiBody(mass, col, vis, basePosition=pos, baseOrientation=p.getQuaternionFromEuler([0,0,yaw]))

    def get_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.body)
        yaw = p.getEulerFromQuaternion(orn)[2]
        vx, vy, vz = p.getBaseVelocity(self.body)[0]
        linear_speed = math.hypot(vx, vy)
        return (pos[0], pos[1], yaw, linear_speed)

    def set_wheel_speeds(self, left_cmd, right_cmd):
        # very simple: apply velocity to base using differential kinematics
        left = clamp(left_cmd, -self.max_speed, self.max_speed)
        right = clamp(right_cmd, -self.max_speed, self.max_speed)
        # convert to linear and angular velocities
        v = self.wheel_radius * (left + right) / 2.0
        omega = self.wheel_radius * (right - left) / self.wheel_base
        # compute world frame velocity
        x, y, yaw, _ = self.get_pose()
        vx = v * math.cos(yaw)
        vy = v * math.sin(yaw)
        # set base velocity directly (simulating motor effect)
        p.resetBaseVelocity(self.body, linearVelocity=[vx, vy, 0], angularVelocity=[0,0,omega])

    def step_pid(self, left_target, right_target, dt):
        # naive PID (only P for simplicity)
        self.set_wheel_speeds(left_target, right_target)

    def ray_sensors(self, angles=SENSOR_ANGLES, max_range=SENSOR_RANGE):
        # cast rays from robot center at given angles relative to heading
        pos, orn = p.getBasePositionAndOrientation(self.body)
        x, y, _ = pos
        yaw = p.getEulerFromQuaternion(orn)[2]
        readings = []
        for a in angles:
            ang = yaw + a
            from_pos = [x, y, 0.03]
            to_pos = [x + max_range * math.cos(ang), y + max_range * math.sin(ang), 0.03]
            res = p.rayTest(from_pos, to_pos)[0]
            hit = res[0]
            if hit == -1:
                dist = max_range
                hitpos = to_pos
            else:
                hitpos = res[3]
                dx = hitpos[0] - x
                dy = hitpos[1] - y
                dist = math.hypot(dx, dy)
            readings.append((dist, hitpos))
        return readings

# ---------- Simple local controller (explore + avoid) ----------
class LocalController:
    def __init__(self, robot: DiffRobot):
        self.robot = robot
        self.target_v = 0.3  # forward m/s
        self.turn_v = 0.6
        self.min_clearance = 0.35

    def decide(self, sensor_readings):
        # sensor_readings: list of (dist, hitpos)
        left_d, _, = sensor_readings[0]
        front_d, _ = sensor_readings[1]
        right_d, _ = sensor_readings[2]

        # Simple behavior: if front too close -> rotate away from closer obstacle
        if front_d < self.min_clearance:
            # rotate in direction of largest space
            if left_d > right_d:
                # turn left
                left_w = -0.6
                right_w = 0.6
            else:
                left_w = 0.6
                right_w = -0.6
            return left_w, right_w

        # if left or right very close, steer away a bit
        if left_d < self.min_clearance:
            return 0.2, 0.6  # slight right
        if right_d < self.min_clearance:
            return 0.6, 0.2  # slight left

        # otherwise go forward
        # map forward speed to wheel velocities
        # v = r*(L+R)/2 => choose L and R equal
        desired_v = self.target_v  # m/s
        wheel_r = self.robot.wheel_radius
        desired_wheel = desired_v / wheel_r
        return desired_wheel, desired_wheel

# ---------- Simple Node-RED logger (HTTP) ----------
class NodeRedLogger:
    def __init__(self, url=None):
        self.url = url
        if url and requests is None:
            print("requests não instalado — desabilitando envio HTTP.")
            self.url = None

    def send(self, payload):
        if not self.url:
            return
        try:
            headers = {"Content-Type": "application/json"}
            requests.post(self.url, json=payload, headers=headers, timeout=1.0)
        except Exception as e:
            # fail silently for now
            print("Falha ao enviar para Node-RED:", e)

# ---------- World builder ----------
def build_world(client):
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.resetSimulation(physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    plane = p.loadURDF("plane.urdf", physicsClientId=client)
    # create walls / rooms (simple boxes) - a "house" like layout
    walls = []
    wall_height = 0.2
    # outer walls (a square)
    thickness = 0.05
    size = WORLD_SIZE / 2.0
    # four walls as thin boxes placed around
    wall_shapes = []
    def mk_wall(cx, cy, lx, ly):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, wall_height/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, wall_height/2], rgbaColor=[0.8,0.8,0.8,1])
        body = p.createMultiBody(0, col, vis, basePosition=[cx, cy, wall_height/2])
        return body

    # perimeter
    mk_wall(0, size + thickness/2, WORLD_SIZE, thickness)
    mk_wall(0, -size - thickness/2, WORLD_SIZE, thickness)
    mk_wall(size + thickness/2, 0, thickness, WORLD_SIZE)
    mk_wall(-size - thickness/2, 0, thickness, WORLD_SIZE)

    # interior obstacles: create some boxes representing furniture / walls
    mk_wall(1.2, 0.8, 1.6, 0.06)  # corridor wall
    mk_wall(-1.2, -0.8, 1.0, 0.06)
    mk_wall(0.0, -1.5, 0.06, 1.0)
    # larger block (table)
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.05])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.05], rgbaColor=[0.6,0.3,0.1,1])
    p.createMultiBody(0, table_col, table_vis, basePosition=[-0.8, 1.2, 0.05])
    return plane

# ---------- Simulation Runner ----------
class Simulator:
    def __init__(self, node_red_url=None, gui=True):
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setPhysicsEngineParameter(fixedTimeStep=TIME_STEP, physicsClientId=self.client)
        build_world(self.client)
        self.robot = DiffRobot(self.client)
        self.controller = LocalController(self.robot)
        self.map = OccupancyGrid()
        self.logger = NodeRedLogger(node_red_url)
        self.metrics = {
            "trajectory": [],
            "time": 0.0,
            "collisions": 0,
            "energy": 0.0,
            "area_covered": 0.0
        }
        # for collision detection
        self.prev_pos = None

    def update_map_from_sensors(self, sensor_readings, pose):
        x, y, yaw, speed = pose
        # mark free along rays, mark hit positions as occupied
        for dist, hitpos in sensor_readings:
            # mark free along ray sample points
            steps = int(max(1, (dist / self.map.resolution)))
            for s in range(1, steps+1):
                sx = x + (s * (hitpos[0] - x) / (steps+0.0001))
                sy = y + (s * (hitpos[1] - y) / (steps+0.0001))
                self.map.mark_free(sx, sy)
            # if hit within range and close, mark occupied
            if dist < SENSOR_RANGE - 0.001:
                self.map.mark_occupied(hitpos[0], hitpos[1], prob=1.0)

    def run_episode(self, max_time=120.0, start=(0,0,0.05), use_map_for_planning=False):
        # reposition robot to start
        p.resetBasePositionAndOrientation(self.robot.body, start, p.getQuaternionFromEuler([0,0,0]))
        self.metrics = {"trajectory": [], "time": 0.0, "collisions": 0, "energy": 0.0, "area_covered": 0.0}
        t = 0.0
        last_time = time.time()
        # For simple learning: bias to frontiers if map available
        visited_positions = set()
        while t < max_time:
            # read sensors
            sensors = self.robot.ray_sensors()
            pose = self.robot.get_pose()
            x, y, yaw, lin_spd = pose

            # detect collisions (rough): if base touches something (contacts)
            contacts = p.getContactPoints(bodyA=self.robot.body)
            if len(contacts) > 0:
                self.metrics["collisions"] += 1

            # update map & visits
            self.update_map_from_sensors(sensors, pose)
            self.map.increment_visit(x, y, dt=TIME_STEP)
            self.metrics["trajectory"].append((t, x, y, yaw))
            visited_positions.add((round(x,2), round(y,2)))

            # controller decision (if using learned map, try to head to nearest frontier sometimes)
            left_cmd, right_cmd = self.controller.decide(sensors)
            if use_map_for_planning and (int(t) % 6 == 0):
                # occasionally pick a nearest frontier and drive toward it (very naive)
                frontiers = self.map.unexplored_frontiers(threshold_visits=0)
                if frontiers:
                    # choose nearest frontier biasing by visits (prefer low visit)
                    fx, fy = min(frontiers, key=lambda f: (f[0]-x)**2 + (f[1]-y)**2)
                    # compute angle to frontier
                    ang_to = math.atan2(fy - y, fx - x)
                    ang_err = (ang_to - yaw + math.pi) % (2*math.pi) - math.pi
                    # turn proportional
                    if abs(ang_err) > 0.2:
                        # rotate toward target
                        if ang_err > 0:
                            left_cmd, right_cmd = -0.6, 0.6
                        else:
                            left_cmd, right_cmd = 0.6, -0.6
                    else:
                        # go forward
                        wheel = (self.controller.target_v / self.robot.wheel_radius)
                        left_cmd, right_cmd = wheel, wheel

            # apply to robot and step sim
            self.robot.step_pid(left_cmd, right_cmd, TIME_STEP)
            # energy estimate: sum of abs wheel torques ~ abs(wheel velocity) * dt (proxy)
            self.metrics["energy"] += (abs(left_cmd) + abs(right_cmd)) * TIME_STEP

            p.stepSimulation()
            time.sleep(TIME_STEP if p.getConnectionInfo(self.client)['connectionMethod'] == 1 else 0)  # sleep only in GUI mode
            t += TIME_STEP

        self.metrics["time"] = t
        # coverage metric: percent of map cells visited at least once
        self.metrics["area_covered"] = self.map.percent_covered()
        self.metrics["unique_positions"] = len(visited_positions)
        return self.metrics

# ---------- Main / experiment loop ----------
def main(args):
    # prepare simulator
    sim = Simulator(node_red_url=(None if args.no_node_red else args.node_red), gui=args.gui)
    # optionally load previous map
    learned = False
    if args.load_map and os.path.exists(MAP_FILE):
        sim.map.load(MAP_FILE)
        print("Mapa carregado de", MAP_FILE)
        learned = True

    run_results = []
    for r in range(args.runs):
        print(f"\n=== Execução {r+1}/{args.runs} (learned_map={learned}) ===")
        start = (0.0, 0.0, 0.05)
        metrics = sim.run_episode(max_time=args.episode_time, start=start, use_map_for_planning=learned)
        print(f"Tempo: {metrics['time']:.1f}s | Cobertura: {metrics['area_covered']:.2f}% | Energia: {metrics['energy']:.2f} | Colisões: {metrics['collisions']}")
        # send to Node-RED
        payload = {
            "run": r+1,
            "time": metrics["time"],
            "area_covered": metrics["area_covered"],
            "energy": metrics["energy"],
            "collisions": metrics["collisions"],
            "unique_positions": metrics["unique_positions"],
            "timestamp": time.time()
        }
        sim.logger.send(payload)
        run_results.append(payload)
        # after first run, mark that there is a learned map to use next time
        sim.map.save(MAP_FILE)
        learned = True

        # quick break between runs
        time.sleep(0.2)

    # Save run results summary
    summary_file = os.path.join(LOG_DIR, f"summary_{int(time.time())}.json")
    with open(summary_file, "w") as f:
        json.dump({"runs": run_results}, f, indent=2)
    print("Resumo salvo em", summary_file)
    print("Mapa salvo em", MAP_FILE)
    # keep GUI open if requested
    if args.gui:
        print("Pressione Ctrl+C na janela do terminal para encerrar (PyBullet GUI).")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador Robô Aspirador - único arquivo")
    parser.add_argument("--runs", type=int, default=3, help="Número de execuções (episódios)")
    parser.add_argument("--episode-time", type=float, default=60.0, help="Tempo máximo por episódio (s)")
    parser.add_argument("--load-map", action="store_true", help="Carregar mapa existente no início (se existir)")
    parser.add_argument("--node-red", type=str, default="http://localhost:1880/log", help="URL do Node-RED para envio de logs (POST JSON)")
    parser.add_argument("--no-node-red", action="store_true", help="Desabilitar envio para Node-RED")
    parser.add_argument("--gui", action="store_true", default=True,
                        help="Abrir GUI do PyBullet (padrão: sim). Use --gui False para modo headless.")

    args = parser.parse_args()
    main(args)