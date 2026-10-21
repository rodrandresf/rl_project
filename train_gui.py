import os
import gymnasium as gym
import numpy as np
from typing import Final, List, Optional
from agents.q_learning import QLearningAgent
from utils.live_plot import LivePlotter

TOTAL_EPISODES: Final[int] = 1000
RENDER_FREQ: Final[int] = 100  # Mostrar la GUI del Taxi cada 100 episodios
GAMMA: Final[float] = 0.95
EPSILON_START: Final[float] = 1.0
EPSILON_DECAY: Final[float] = 0.995

def run_visual_experiment() -> None:
    # Entorno base sin renderizado (procesamiento en RAM)
    env_fast: gym.Env = gym.make("Taxi-v3")
    # Entorno con GUI activa
    env_gui: gym.Env = gym.make("Taxi-v3", render_mode="human")
    
    agent: QLearningAgent = QLearningAgent(
        action_size=int(env_fast.action_space.n), 
        state_size=int(env_fast.observation_space.n)
    )
    
    plotter: LivePlotter = LivePlotter(window_size=20)
    rewards_history: List[float] = []
    epsilon: float = EPSILON_START

    for episode in range(TOTAL_EPISODES):
        # Determinamos si este episodio debe ser renderizado
        is_render_episode: bool = (episode % RENDER_FREQ == 0) or (episode == TOTAL_EPISODES - 1)
        current_env = env_gui if is_render_episode else env_fast
        
        state, _ = current_env.reset()
        terminated: bool = False
        truncated: bool = False
        total_reward: float = 0.0

        if is_render_episode:
            print(f"--- Evaluando visualmente Episodio {episode} | Epsilon: {epsilon:.2f} ---")

        while not (terminated or truncated):
            action: int = agent.choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = current_env.step(action)
            
            agent.update(state, action, float(reward), next_state, GAMMA)
            
            state = next_state
            total_reward += float(reward)

        epsilon = max(0.01, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)

        # Actualizamos la gráfica cada 10 episodios para no ahogar el procesador
        if episode % 10 == 0:
            plotter.update(rewards_history)

    env_fast.close()
    env_gui.close()
    
    os.makedirs("models", exist_ok=True)
    np.save("models/q_table_gui.npy", agent.q_table)

if __name__ == "__main__":
    run_visual_experiment()