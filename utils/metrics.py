import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def plot_learning_curve(rewards: List[float], window: int = 50) -> None:
    """
    Genera un gráfico de la evolución de las recompensas.
    Usa una media móvil (window) para suavizar el ruido.
    """
    plt.figure(figsize=(10, 5))
    
    # Cálculo vectorizado de la media móvil
    if len(rewards) >= window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smooth_rewards, label=f"Media Móvil ({window} eps)")
    
    plt.plot(rewards, alpha=0.3, label="Recompensa por Episodio", color='gray')
    plt.title("Progreso del Agente (Curva de Aprendizaje)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Acumulada")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png") # Guarda el resultado automáticamente
    plt.show()
    # ... tu código gráfico anterior ...
    pass


def export_metrics_to_parquet(rewards: List[float], filepath: str = "logs/training_metrics.parquet") -> None:
    """
    Persistencia columnar de métricas de entrenamiento.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df: pd.DataFrame = pd.DataFrame({
        "episode": range(len(rewards)),
        "reward": rewards
    })
    
    df.to_parquet(filepath, engine="pyarrow", compression="snappy")