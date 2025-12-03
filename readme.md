# 🤖 Differential Robot Coverage Simulation

Simulação avançada de **robótica móvel autônoma**, combinando
controle PID, sensores ultrassônicos, física 3D realista e supervisão
IoT em tempo real via **Node-RED + MQTT**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Node-RED](https://img.shields.io/badge/IoT-Node--RED-8F0000?logo=node-red&logoColor=white)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-orange)
![MQTT](https://img.shields.io/badge/Protocol-MQTT-660066)
![Status](https://img.shields.io/badge/Status-Completed-success)

------------------------------------------------------------------------

## 📖 Visão Geral

Este projeto implementa um sistema completo de **exploração autônoma** 
com robô diferencial, capaz de:

- Evitar obstáculos usando sensores ultrassônicos.
- Explorar áreas desconhecidas e registrar cobertura do ambiente.
- Controlar as rodas com estabilidade usando PID.
- Publicar **telemetria em tempo real** para dashboards via **MQTT/Node-RED**.

A simulação combina física realista (PyBullet), controle PID e transmissão
contínua de dados para supervisório.

------------------------------------------------------------------------

## 🎯 Funcionalidades Principais

### 🧠 Navegação Inteligente

- Desvio de obstáculos baseado em leitura de sensores ultrassônicos.
- Exploração guiada por mapa de cobertura, buscando áreas menos visitadas.

### 🎛 Controle PID

- PIDs independentes para as rodas esquerda e direita.
- Controle de torque com limites máximos e ruído simulado.

### 🌀 Física e Realismo

- Simulação completa da física do robô no PyBullet.
- Sensores com ruído adicionado para maior realismo.

### 📡 Dashboard IoT em Tempo Real

- Telemetria completa: posição, odometria, velocidade, cobertura.
- Dados publicados no tópico MQTT `robot/simulation`.
- Visualização possível no Node-RED com dashboards customizados.

------------------------------------------------------------------------

## 📂 Estrutura do Projeto

    differential_robot_sim/
    │
    ├── main.py
    ├── robot.py
    ├── controller.py
    ├── coverage_map.py
    ├── environment.py
    ├── requirements.txt
    └── assets/


------------------------------------------------------------------------

## 🛠️ Pré-Requisitos

- **Python 3.10+**
- **PyBullet**
- **NumPy**
- **Paho-MQTT**
- **Node-RED**
- **Mosquitto MQTT Broker**

------------------------------------------------------------------------

## 🚀 Instalação

### 1️⃣ MQTT Broker
Verifique se o Mosquitto está rodando:
- Windows: Serviços → Mosquitto Broker → Iniciado
- Docker: use a configuração mínima no `mosquitto.conf`

listener 1883
allow_anonymous true


### 2️⃣ Ambiente Python
Abra o terminal na pasta do projeto:

python -m venv venv

Linux/Mac

source venv/bin/activate

Windows

venv\Scripts\activate
pip install -r requirements.txt


### 3️⃣ Node-RED

- Instalar `node-red-dashboard`
- Criar dashboard e importar tópicos MQTT
- Se usar Docker → MQTT host: `host.docker.internal`

### 4️⃣ Executar Simulação


------------------------------------------------------------------------

## 📊 O que aparece?

- Janela PyBullet com o robô diferencial explorando o ambiente.
- Logs de telemetria no terminal.
- Dashboard Node-RED com mapa de cobertura e gráficos de sensores.

------------------------------------------------------------------------

## 🐛 Troubleshooting

| Problema                | Solução                                                       |
|--------------------------|---------------------------------------------------------------|
| Robô não se move         | Verificar torque máximo e PID limits                          |
| Sensores retornam `inf`  | Ajustar `SENSOR_RANGE` ou posição dos sensores               |
| Dashboard Node-RED vazio | Confirmar MQTT broker ativo e tópico correto (`robot/simulation`) |
| Simulação lenta          | Desativar `SIM_REALTIME` ou reduzir `SIM_TIMESTEP`           |

------------------------------------------------------------------------

## 📜 Licença

Uso educacional livre.