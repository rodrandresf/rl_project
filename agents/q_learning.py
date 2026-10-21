from typing import Tuple
import numpy as np

class QLearningAgent:
    def __init__(self, action_size: int, state_size: int, lr: float = 0.1):
        # Usamos float32 para ahorrar memoria
        self.q_table = np.zeros((state_size, action_size), dtype=np.float32)
        self.lr = lr

    def choose_action(self, state: int, epsilon: float) -> int:
        """
        Estrategia Epsilon-Greedy:
        Decide si explorar (azar) o explotar (conocimiento).
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        # Explotación: tomar la acción con mayor valor Q para ese estado
        return int(np.argmax(self.q_table[state]))

    def update(self, s: int, a: int, r: float, s_next: int, gamma: float) -> None:
        """ Actualización de Bellman vectorizada """
        best_next_q = np.max(self.q_table[s_next])
        td_target = r + gamma * best_next_q
        self.q_table[s, a] += self.lr * (td_target - self.q_table[s, a])