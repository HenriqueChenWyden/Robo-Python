import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import json
import os
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

# ==========================================
# 1. CONFIGURAÇÕES E CONSTANTES
# ==========================================
USE_GUI = True              # True para ver a simulação, False para rodar mais rápido
MQTT_BROKER = "localhost"   # IP do Node-RED
MQTT_TOPIC = "robo/telemetria"
MAP_FILE = "mapa_memoria.npy"
ROBOT_URDF = "roomba.urdf"

# Configuração do Grid (Mapa Interno)
MAP_SIZE_METERS = 12.0      # Espaço total de 12x12m
RESOLUTION = 0.2            # Cada célula tem 20cm
GRID_DIM = int(MAP_SIZE_METERS / RESOLUTION)

# Códigos do Mapa
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OBSTACLE = 2
CELL_VISITED = 5            # Área limpa

# ==========================================
# 2. GERADOR DE ROBÔ (URDF)
# ==========================================
def create_roomba_urdf():
    """Gera o arquivo XML do robô com estabilidade (Caster Frontal e Traseiro)"""
    urdf_content = """<?xml version="1.0"?>
<robot name="roomba_robot">
  
  <material name="white"><color rgba="0.9 0.9 0.9 1"/></material>
  <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
  <material name="blue"><color rgba="0 0.5 1 1"/></material>

  <link name="base_link">
    <visual>
      <geometry><cylinder length="0.08" radius="0.17"/></geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.08" radius="0.17"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <link name="led">
    <visual>
      <geometry><box size="0.02 0.05 0.02"/></geometry>
      <material name="blue"/>
    </visual>
  </link>
  <joint name="led_joint" type="fixed">
    <parent link="base_link"/>
    <child link="led"/>
    <origin xyz="0.15 0 0.045"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry><cylinder length="0.02" radius="0.033"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.033"/></geometry>
      <surface>
        <friction>
          <ode><mu>100.0</mu><mu2>100.0</mu2></ode>
        </friction>
      </surface>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin rpy="-1.5707 0 0" xyz="0 0.14 -0.025"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry><cylinder length="0.02" radius="0.033"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry><cylinder length="0.02" radius="0.033"/></geometry>
      <surface>
        <friction>
          <ode><mu>100.0</mu><mu2>100.0</mu2></ode>
        </friction>
      </surface>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="right_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin rpy="-1.5707 0 0" xyz="0 -0.14 -0.025"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="front_caster">
    <visual>
      <geometry><sphere radius="0.032"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry><sphere radius="0.032"/></geometry>
      <surface>
        <friction>
          <ode><mu>0.0</mu><mu2>0.0</mu2></ode> </friction>
      </surface>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  
  <joint name="front_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="front_caster"/>
    <origin xyz="0.13 0 -0.025"/> </joint>

  <link name="rear_caster">
    <visual>
      <geometry><sphere radius="0.032"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry><sphere radius="0.032"/></geometry>
       <surface>
        <friction>
          <ode><mu>0.0</mu><mu2>0.0</mu2></ode>
        </friction>
      </surface>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  
  <joint name="rear_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="rear_caster"/>
    <origin xyz="-0.13 0 -0.025"/>
  </joint>

</robot>
"""
    with open(ROBOT_URDF, "w") as f:
        f.write(urdf_content)
    print(f"URDF Atualizado: {ROBOT_URDF}")

# ==========================================
# 3. CLASSE DE CONTROLE E MEMÓRIA
# ==========================================
class RobotBrain:
    def __init__(self):
        # MQTT Setup
        self.client = mqtt.Client(client_id="Robot_Sim_V1")
        self.mqtt_ok = False
        try:
            self.client.connect(MQTT_BROKER, 1883, 60)
            self.client.loop_start()
            self.mqtt_ok = True
            print("[IoT] MQTT Conectado.")
        except:
            print("[IoT] Aviso: MQTT offline. Rodando local.")

        # Gerenciamento de Memória/Mapa
        self.start_time = time.time()
        self.energy = 0.0
        self.visited_count = 0
        
        # Carrega mapa se existir
        if os.path.exists(MAP_FILE):
            print("[SISTEMA] Memória encontrada! Modo: OTIMIZAÇÃO.")
            self.grid = np.load(MAP_FILE)
            self.learning_mode = True
        else:
            print("[SISTEMA] Memória vazia. Modo: EXPLORAÇÃO.")
            self.grid = np.zeros((GRID_DIM, GRID_DIM), dtype=int)
            self.learning_mode = False

    def world_to_grid(self, x, y):
        # Transforma coord mundo (metros) em coord matriz (índices)
        gx = int((x + MAP_SIZE_METERS/2) / RESOLUTION)
        gy = int((y + MAP_SIZE_METERS/2) / RESOLUTION)
        return gx, gy

    def update_map(self, x, y, sensor_hits):
        # 1. Marcar posição atual como visitada
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM:
            current_val = self.grid[gx][gy]
            if current_val != CELL_VISITED and current_val != CELL_OBSTACLE:
                self.grid[gx][gy] = CELL_VISITED
                self.visited_count += 1
        
        # 2. Marcar Obstáculos (Paredes detectadas)
        for hx, hy in sensor_hits:
            ix, iy = self.world_to_grid(hx, hy)
            if 0 <= ix < GRID_DIM and 0 <= iy < GRID_DIM:
                self.grid[ix][iy] = CELL_OBSTACLE

    def get_cell_value(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < GRID_DIM and 0 <= gy < GRID_DIM:
            return self.grid[gx][gy]
        return CELL_OBSTACLE # Trata fora do mapa como parede

    def send_telemetry(self, x, y):
        if not self.mqtt_ok: return
        
        elapsed = time.time() - self.start_time
        efficiency = (self.visited_count * (RESOLUTION**2)) / (self.energy + 0.1)
        
        payload = {
            "x": round(x, 2), "y": round(y, 2),
            "energy": round(self.energy, 2),
            "area_covered": int(self.visited_count),
            "efficiency": round(efficiency, 4),
            "mode": "Learning" if self.learning_mode else "Exploration"
        }
        self.client.publish(MQTT_TOPIC, json.dumps(payload))

    def save_memory(self):
        np.save(MAP_FILE, self.grid)
        print("[SISTEMA] Mapa salvo em disco.")

# ==========================================
# 4. FUNÇÕES DO PYBULLET
# ==========================================
def get_sensors(robot_id):
    """Lança raios e retorna distâncias (Corrigido para não bater no próprio robô)"""
    pos, orient = p.getBasePositionAndOrientation(robot_id)
    yaw = p.getEulerFromQuaternion(orient)[2]
    
    # Configuração dos Sensores
    # [Angulo relativo (rad), Alcance (m)]
    sensor_config = [
        (0, 2.5),       # Frente
        (0.5, 1.5),     # Esquerda Diag
        (-0.5, 1.5),    # Direita Diag
        (1.2, 1.0),     # Esq Lateral (quase 90 graus)
        (-1.2, 1.0)     # Dir Lateral
    ]
    
    distances = []
    hit_points = []
    
    # --- CORREÇÃO AQUI ---
    # O raio sai 20cm do centro (o robô tem raio 17cm), então sai 3cm "no ar" fora do corpo
    ray_start_radius = 0.20 
    # Altura relativa: Pega a altura atual do robô (pos[2]) e sobe 5cm
    sensor_z = pos[2] + 0.05 
    
    for angle, r_range in sensor_config:
        ray_angle = yaw + angle
        
        # Ponto Inicial (Fora do corpo do robô)
        start = [
            pos[0] + ray_start_radius * math.cos(ray_angle),
            pos[1] + ray_start_radius * math.sin(ray_angle),
            sensor_z
        ]
        # Ponto Final
        end = [
            pos[0] + (ray_start_radius + r_range) * math.cos(ray_angle),
            pos[1] + (ray_start_radius + r_range) * math.sin(ray_angle),
            sensor_z
        ]
        
        # Lança o raio
        res = p.rayTest(start, end)[0]
        hit_id = res[0]
        hit_pos = res[3]
        
        # Se bateu em algo (E ESSE ALGO NÃO É O PRÓPRIO ROBÔ)
        if hit_id != -1 and hit_id != robot_id: 
            dist = np.linalg.norm(np.array(start) - np.array(hit_pos))
            distances.append(dist)
            hit_points.append(hit_pos[:2])
            p.addUserDebugLine(start, hit_pos, [1, 0, 0], 1, 0.1) # Vermelho = Obstáculo
        else:
            distances.append(r_range)
            p.addUserDebugLine(start, end, [0, 1, 0], 1, 0.1) # Verde = Livre
            
    return distances, hit_points, pos, yaw

def setup_simulation():
    create_roomba_urdf()
    
    if USE_GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # Criar Paredes (Box Shapes)
    wall_opts = [
        ([6, 0.5, 1], [0, 6, 0.5]),  # Norte
        ([6, 0.5, 1], [0, -6, 0.5]), # Sul
        ([0.5, 6, 1], [6, 0, 0.5]),  # Leste
        ([0.5, 6, 1], [-6, 0, 0.5]), # Oeste
        ([0.5, 2, 1], [0, 0, 0.5]),  # Obstaculo Central
        ([1, 1, 1], [-3, -3, 0.5]),  # Obstaculo Canto
    ]
    
    for dim, pos in wall_opts:
        vid = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, rgbaColor=[0.5,0.3,0.1,1])
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
        p.createMultiBody(0, cid, vid, basePosition=pos)
        
    # Carregar Robô
    rid = p.loadURDF(ROBOT_URDF, [0, -4, 0.1])
    return rid

# ==========================================
# 5. LOOP PRINCIPAL
# ==========================================
def main():
    robot_id = setup_simulation()
    brain = RobotBrain()
    
    # Parâmetros de Navegação
    BASE_SPEED = 30.0
    TURN_SPEED = 30.0
    
    print(">>> Simulação Iniciada. Pressione Ctrl+C para parar e salvar.")
    
    try:
        # Loop Infinito (ou até fechar GUI)
        while p.isConnected():
            p.stepSimulation()
            
            # 1. Ler Sensores
            # dists indices: 0=Frente, 1=EsqFrontal, 2=DirFrontal, 3=EsqLat, 4=DirLat
            dists, hits, pos, yaw = get_sensors(robot_id)
            
            # 2. Atualizar Cérebro
            brain.update_map(pos[0], pos[1], hits)
            
            # 3. Lógica de Decisão
            vl, vr = BASE_SPEED, BASE_SPEED
            
            # Verificar futuro imediato (Aprendizado)
            look_ahead = 0.6
            next_x = pos[0] + look_ahead * math.cos(yaw)
            next_y = pos[1] + look_ahead * math.sin(yaw)
            cell_ahead = brain.get_cell_value(next_x, next_y)
            
            # Lógica de Prioridade:
            # 1. Evasão de Colisão (Reativa)
            if dists[0] < 0.6 or dists[1] < 0.4 or dists[2] < 0.4:
                # Obstáculo perto -> Girar
                # Decide lado baseado nos sensores laterais
                if dists[3] > dists[4]: # Mais espaço na esquerda
                    vl, vr = -TURN_SPEED, TURN_SPEED
                else:
                    vl, vr = TURN_SPEED, -TURN_SPEED
            
            # 2. Otimização de Rota (Cognitiva - Só se estiver aprendendo)
            elif brain.learning_mode and cell_ahead == CELL_VISITED:
                # Se à frente está limpo, desvia suavemente para tentar achar sujeira
                # Isso cria um comportamento de "varredura de borda"
                vl = BASE_SPEED * 0.5
                vr = BASE_SPEED * 0,5
            
            # Tente inverter o sinal da roda direita se ele estiver girando no eixo
            p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=-vl, force=135)
            p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=-vr, force=135) # Note o MENOS
            
            # 5. Contabilizar Energia
            brain.energy += (abs(vl) + abs(vr)) * 0.001
            
            # 6. Telemetria (Reduzir frequencia de envio)
            if int(time.time() * 10) % 5 == 0: # A cada ~0.5s
                brain.send_telemetry(pos[0], pos[1])
                
            time.sleep(1./240.)

    except KeyboardInterrupt:
        pass
    except p.error:
        print("Janela fechada.")
        
    # Encerramento
    print(f"\n--- FIM ---")
    print(f"Eficiência: {brain.visited_count / (brain.energy+1):.2f}")
    brain.save_memory()
    p.disconnect()
    
    # Mostrar Mapa Final
    plt.figure(figsize=(6,6))
    plt.imshow(brain.grid, cmap='terrain', origin='lower')
    plt.title(f"Mapa de Ocupação\nModo: {'Aprendizado' if brain.learning_mode else 'Exploração'}")
    plt.colorbar(label="0=Desc, 2=Par, 5=Limp")
    plt.show()

if __name__ == "__main__":
    main()