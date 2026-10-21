import gymnasium as gym
import numpy as np
import os
from agents.q_learning import QLearningAgent
from typing import Final

TOTAL_EPISODES: Final = 10  # Solo 10 para la prueba rápida
GAMMA: Final = 0.95

def run_experiment() -> None:
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(env.action_space.n, env.observation_space.n)
    
    print("🚀 Iniciando prueba de integración...")

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Acción aleatoria para la prueba
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Validamos que el agente procese la actualización
            agent.update(state, action, reward, next_state, GAMMA)
            state = next_state

        print(f"✅ Episodio {episode} completado.")

    # Persistencia
    os.makedirs("models", exist_ok=True)
    np.save("models/q_table_test.npy", agent.q_table)
    print("\n¡ÉXITO! El agente puede interactuar, aprender y guardar datos.")

if __name__ == "__main__":
    run_experiment()