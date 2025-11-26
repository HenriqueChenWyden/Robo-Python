# Robô Aspirador Inteligente (Simulação PyBullet + Node-RED)
Este projeto simula um robô aspirador capaz de mapear o ambiente, evitar
obstáculos, gerar mapas e enviar telemetria em tempo real.
## Features
- Base diferencial simulada em PyBullet
- Sensores ultrassônicos via Raycast
- Mapeamento 2D (Occupancy Grid)
- Aprendizado por repetição (salva e reutiliza mapa)
- Logs via MQTT para Node-RED
11
## Execução
```bash
# Requisitos
É necessário usar **Python 3.12.2**.
```bash
pip install -r requirements.txt
python main.py