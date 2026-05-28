# @author: Íñigo Martínez Jiménez
# This module defines the vectorized evaluator used in the PSO.
# Instead of calling the fitness function once per particle (n_particles Python
# calls), it calls a single NumPy-native function that processes the full
# position matrix at once. All arithmetic runs in compiled C code without
# Python-level loops, which is the "implicit parallelism" of NumPy.

from collections.abc import Callable

import numpy as np


class V4_vectorized:
    """
    Vectorized evaluator for particle fitness values.

    Requires a vectorized objective function with signature:
    (np.ndarray of shape (n_particles, dim)) -> (np.ndarray of shape (n_particles,))

    The speedup over V0 comes from eliminating the Python loop: one BLAS/C call
    replaces n_particles individual Python to NumPy transitions.
    """

    def __init__(self, vectorized_f: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Initialize the evaluator with a vectorized objective function.

        Args:
            vectorized_f (Callable[[np.ndarray], np.ndarray]): Objective function
                that accepts a (n_particles, dim) matrix and returns a (n_particles,)
                array. See pso/objectives/vectorized_functions.py for implementations.
        """
        self.vectorized_f = vectorized_f

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of all particles in one vectorized call.

        Args:
            positions (np.ndarray): Position matrix of shape (n_particles, dim).

        Returns:
            np.ndarray: Fitness values of shape (n_particles,).
        """
        # Single call: no Python loop, all computation happens inside NumPy/C
        return self.vectorized_f(positions).astype(float)
    
    def shutdown(self) -> None:
        """Release evaluator resources. No-op for the vectorized evaluator."""
    pass