# train.py (Cabecera corregida)
import os
import gymnasium as gym
import numpy as np
from agents.q_learning import QLearningAgent

# SOLUCIÓN: La importación debe incluir ambas funciones separadas por coma
from utils.metrics import plot_learning_curve, export_metrics_to_parquet

# Hiperparámetros de producción
from typing import Final, List

# Hiperparámetros optimizados para convergencia total
TOTAL_EPISODES: Final[int] = 5000       # Incremento de iteraciones para mapeo completo
GAMMA: Final[float] = 0.99              # Visión a largo plazo casi absoluta
EPSILON_START: Final[float] = 1.0
EPSILON_DECAY: Final[float] = 0.999     # Decaimiento lento: obliga a explorar miles de veces
MIN_EPSILON: Final[float] = 0.01        # Límite mínimo de estocasticidad

def run_experiment() -> None:
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(env.action_space.n, env.observation_space.n)
    
    rewards_history: List[float] = []
    epsilon = EPSILON_START

    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            # Ahora el agente elige basándose en su aprendizaje
            action = agent.choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, GAMMA)
            
            state = next_state
            total_reward += reward

        # Reducimos la exploración gradualmente
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)

        if episode % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            print(f"Eps: {episode} | Avg Reward: {avg_r:.1f} | Epsilon: {epsilon:.2f}")

    env.close()

    # Persistencia de artefactos (Modelo)
    os.makedirs("models", exist_ok=True)
    np.save("models/q_table_final.npy", agent.q_table)
    
    # Persistencia de telemetría columnar (Métricas)
    export_metrics_to_parquet(rewards_history)
    
    # Visualización final
    plot_learning_curve(rewards_history)

if __name__ == "__main__":
    run_experiment()