import matplotlib.pyplot as plt
import numpy as np
from typing import List

class LivePlotter:
    """
    Gestor de visualización en tiempo real no bloqueante para métricas de RL.
    """
    def __init__(self, window_size: int = 50) -> None:
        plt.ion() # Activa el modo interactivo
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.window_size = window_size
        self.line_reward, = self.ax.plot([], [], alpha=0.3, color='gray', label='Recompensa/Episodio')
        self.line_avg, = self.ax.plot([], [], color='#1f77b4', linewidth=2, label=f'Media Móvil ({window_size} eps)')
        
        self.ax.set_title("Evolución del Agente en Tiempo Real")
        self.ax.set_xlabel("Episodio")
        self.ax.set_ylabel("Recompensa Acumulada")
        self.ax.legend(loc="lower right")
        self.ax.grid(True, linestyle='--', alpha=0.6)

    def update(self, rewards_history: List[float]) -> None:
        """
        Actualiza los vectores de datos de la gráfica O(1) operaciones I/O.
        """
        x_data = np.arange(len(rewards_history))
        self.line_reward.set_data(x_data, rewards_history)

        if len(rewards_history) >= self.window_size:
            # Cálculo vectorizado de media móvil
            kernel = np.ones(self.window_size, dtype=np.float32) / self.window_size
            smooth_rewards = np.convolve(rewards_history, kernel, mode='valid')
            x_smooth = np.arange(self.window_size - 1, len(rewards_history))
            self.line_avg.set_data(x_smooth, smooth_rewards)

        # Ajuste dinámico de ejes
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()