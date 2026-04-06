# @author: Íñigo Martínez Jiménez

"""
This module defines the Result data container used to store the output of a
PSO execution. The class groups together the final best solution found by the 
optimizer, the most important timing metrics, the convergence history across
iterations, and the sequence of global best positions recorded during the run.
"""

from dataclasses import dataclass
import numpy as np


# We use @dataclass because automatically generates useful methods such as 
# __init__, __repr__, and this class is only meant to store data
@dataclass
class Result:
    """
    Store the output of a PSO run.

    Args:
        b_position: Best position found by the swarm at the end of the run.
        b_value: Best objective function value found during the optimization.
        total_time: Total execution time of the PSO run.
        fitness_eval_time_total: Total time spent evaluating the objective function across the whole run.
        fitness_eval_time_by_iter: Execution time spent on fitness evaluation at each iteration.
        best_fitness_by_iter: History of the global best fitness value after each iteration.
        iterations: Number of iterations actually executed.
        best_positions_by_iter: History of the global best position at each iteration.
    """
    b_position: np.ndarray
    b_value: float
    total_time: float
    fitness_eval_time_total: float
    fitness_eval_time_by_iter: list[float]
    best_fitness_by_iter: list[float]
    iterations: int
    best_positions_by_iter: list[np.ndarray] | None