from typing import List
import matplotlib.pyplot as plt
import numpy as np

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