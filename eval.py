import time
from typing import Final
import gymnasium as gym
import numpy as np

# Corrección estricta de la ruta del artefacto
MODEL_PATH: Final[str] = "models/q_table_final.npy"  # <-- EL ERROR ESTABA AQUÍ
TEST_EPISODES: Final[int] = 5
LATENCY_DELAY: Final[float] = 0.3

def ejecutar_inferencia_estricta() -> None:
    try:
        # Carga matricial C-contigua O(1)
        q_table: np.ndarray = np.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Falla de pipeline: No se encuentra el artefacto en {MODEL_PATH}")

    env: gym.Env = gym.make("Taxi-v3", render_mode="human")

    for episode in range(TEST_EPISODES):
        state, _ = env.reset()
        terminated: bool = False
        truncated: bool = False
        steps: int = 0
        total_reward: float = 0.0

        while not (terminated or truncated) and steps < 200:
            # Explotación determinista absoluta
            action: int = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += float(reward)
            steps += 1
            time.sleep(LATENCY_DELAY)

        print(f"Viaje {episode + 1} finalizado | Pasos: {steps} | Retorno: {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    ejecutar_inferencia_estricta()