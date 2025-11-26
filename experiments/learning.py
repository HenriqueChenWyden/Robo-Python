import pickle
import random

Q_FILE = "q_table.pkl"

# parâmetros
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# carregar ou criar Q-table
try:
    with open(Q_FILE, "rb") as f:
        Q = pickle.load(f)
except:
    Q = {}

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
