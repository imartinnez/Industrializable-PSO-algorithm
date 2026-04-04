from dataclasses import dataclass
import numpy as np

@dataclass
class Result:
    b_position: np.ndarray
    b_value: float
    total_time: float
    fitness_eval_time_total: float
    fitness_eval_time_by_iter: list[float]
    best_fitness_by_iter: list[float]
    iterations: int
    trajectories: list[np.ndarray] | None # una (N, dim) por iteración
    best_positions_by_iter: list[np.ndarray] | None # el gbest de cada iteración