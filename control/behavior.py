import numpy as np
from typing import Sequence, Tuple


def explore_step(readings: Sequence[float], base_speed: float = 0.25, avoid_thresh: float = 0.6) -> Tuple[float, float]:
    """Decide velocidades (left, right) dadas as leituras de distância.

    Agora reage se qualquer leitura estiver abaixo do limiar (`avoid_thresh`)
    (não somente a frente). Escolhe o lado com mais espaço. Se ambos os
    lados estiverem bloqueados, recua.

    readings: [left, front, right]
    Retorna (left_vel, right_vel) em m/s (velocidades lineares das rodas).
    """
    if len(readings) < 3:
        return base_speed, base_speed

    left = float(readings[0])
    front = float(readings[1])
    right = float(readings[2])

    # parâmetros de manobra
    rotate_speed = 0.30  # velocidade das rodas para girar (m/s)
    backup_speed = -0.2  # velocidade de ré quando preso
    side_margin = 0.05  # diferença mínima para preferir um lado

    # debug opcional via variável de ambiente
    try:
        import os

        if os.environ.get("BEHAVIOR_DEBUG", "0") == "1":
            print(f"[behavior] readings L/F/R = {left:.3f}/{front:.3f}/{right:.3f}")
    except Exception:
        pass

    min_dist = min(left, front, right)

    # caso normal: nenhuma leitura baixa
    if min_dist >= avoid_thresh:
        return base_speed, base_speed

    # se houver obstáculo, priorizar o lado com mais espaço
    # considerar diferença mínima (side_margin)
    if left - right > side_margin and left > 0.12:
        return -rotate_speed, rotate_speed
    if right - left > side_margin and right > 0.12:
        return rotate_speed, -rotate_speed

    # se ambos os lados estiverem muito próximos -> recuar
    if left < avoid_thresh and right < avoid_thresh:
        return backup_speed, backup_speed

    # fallback: gira para o lado com mais espaço
    if left >= right:
        return -rotate_speed, rotate_speed
    return rotate_speed, -rotate_speed