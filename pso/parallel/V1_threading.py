# @author: Íñigo Martínez Jiménez

"""
This module defines the threaded evaluator used in the PSO.
It evaluates the fitness of all particles in parallel using threads.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class V1_threading:
    """
    Thread-based evaluator for particle fitness values.
    """

    def __init__(self, fitness_f: Callable[[np.ndarray], float], max_workers: int | None = None) -> None:
        """
        Initialize the evaluator with the objective function and the number of threads.

        Args:
            fitness_f (Callable[[np.ndarray], float]): Objective function used to evaluate each particle.
            max_workers (int | None): Number of worker threads. If None, Python chooses it automatically.
        """
        self.fitness_f = fitness_f
        self.max_workers = max_workers

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of all particles using threads.

        Args:
            positions (np.ndarray): Positions of all particles.

        Returns:
            np.ndarray: Fitness value of each particle.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            values = list(executor.map(self.fitness_f, positions))

        return np.array(values, dtype=float)