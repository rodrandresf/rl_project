from typing import Tuple
import numpy as np

class QLearningAgent:
    def __init__(self, action_size: int, state_size: int, lr: float = 0.01):
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self.lr = lr

    def update(self, s: int, a: int, r: float, s_next: int, gamma: float):
        # Implementación vectorizada de Bellman
        best_next = np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (r + gamma * best_next - self.q_table[s, a])