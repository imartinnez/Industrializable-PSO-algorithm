# @author: Íñigo Martínez Jiménez
# This module defines the multiprocessing evaluator used in the PSO.
# evaluating the fitness of all particles in parallel using separate processes.


from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
import numpy as np


class V2_multiprocessing:
    """
    Process-based evaluator for particle fitness values.
    """

    def __init__(self, fitness_f: Callable[[np.ndarray], float], max_workers: int | None = None, chunksize: int = 5) -> None:
        """
        Initialize the evaluator with the objective function and multiprocessing settings.

        Args:
            fitness_f (Callable[[np.ndarray], float]): Objective function used to evaluate each particle.
            max_workers (int | None): Number of worker processes. If None, Python chooses it automatically.
            chunksize (int): Number of tasks sent together to each worker process.
        """
        self.fitness_f = fitness_f
        self.chunksize = chunksize
        self._executor = ProcessPoolExecutor(max_workers=max_workers)  # ← UNA sola vez


    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        
        values = list(self._executor.map(self.fitness_f, positions, chunksize=self.chunksize))
        return np.array(values, dtype=float)
    
    def shutdown(self):
        self._executor.shutdown(wait=True)
