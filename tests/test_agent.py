# test_agent.py
import numpy as np
from agents.q_learning import QLearningAgent

def test_q_learning_update():
    # Setup: Agente con 2 acciones y 2 estados
    agent = QLearningAgent(action_size=2, state_size=2, lr=1.0)
    
    # Simulación: Estado 0, Acción 0, Recompensa 10, Siguiente Estado 1
    # Q(0,0) inicial es 0
    agent.update(state=0, action=0, reward=10.0, s_next=1, gamma=0.9)
    
    # Con lr=1.0, Q(0,0) debería ser exactamente 10.0
    assert agent.q_table[0, 0] == 10.0
    print("Test de actualización: PASADO")

if __name__ == "__main__":
    test_q_learning_update()