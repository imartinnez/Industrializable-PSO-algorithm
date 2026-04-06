# @author: Íñigo Martínez Jiménez

"""
This module defines the sequential evaluator used in the PSO.
It evaluates the fitness of all particles one by one, without using parallelism.
"""

import numpy as np


class VO_sequential:
    """
    Sequential evaluator for particle fitness values.
    """

    def __init__(self, fitness_f) -> None:
        """
        Initialize the evaluator with the objective function.

        Args:
            fitness_f: Objective function used to evaluate each particle.
        """
        self.fitness_f = fitness_f

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of all particles sequentially.

        Args:
            positions (np.ndarray): Positions of all particles.

        Returns:
            np.ndarray: Fitness value of each particle.
        """
        return np.array([self.fitness_f(position) for position in positions])