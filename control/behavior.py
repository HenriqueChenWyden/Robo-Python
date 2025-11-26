import numpy as np

def explore_step(readings, base_speed=0.2,
avoid_thresh=0.4):
    front = readings[1]
    left = readings[0]
    right = readings[2]
    
    if front < avoid_thresh:
        if left > right:
            return -0.1, 0.2 # gira esquerda
        else:
            return 0.2, -0.1 # gira direita
    return base_speed, base_speed