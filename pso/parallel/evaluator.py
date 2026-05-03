# @author: Íñigo Martínez Jiménez
# This module defines the evaluator selector used in the PSO.
# It centralizes the logic for choosing how particle fitness is evaluated, so the
# PSO engine does not need to be modified every time a new evaluator is added

from collections.abc import Callable
from typing import Any

from pso.parallel.V0_sequential import VO_sequential
from pso.parallel.V1_threading import V1_threading
from pso.parallel.V2_multiprocessing import V2_multiprocessing
from pso.parallel.V3_async import V3_async
from pso.parallel.V4_vectorized import V4_vectorized


def choose_evaluator(mode: str, fitness_f: Callable,
                     vectorized_f: Callable | None = None,
                     latency_range: tuple[float, float] = (0.005, 0.02)) -> Any:
    """
    Return the evaluator associated with the selected execution mode.

    Args:
        mode (str): Evaluation mode to use.
        fitness_f (Callable): Scalar objective function (1D array → float).
        vectorized_f (Callable | None): Vectorized objective function
            (2D matrix → 1D array). Only used when mode is 'vectorized'.
        latency_range (tuple[float, float]): Simulated I/O latency range in seconds.
            Only used when mode is 'async'.

    Raises:
        ValueError: If the selected mode is not valid.
        ValueError: If mode is 'vectorized' and no vectorized_f is provided.

    Returns:
        Any: Evaluator object used to compute fitness values.
    """
    # This function isolates the evaluator selection logic from the PSO engine.
    # New evaluators can be added here without changing the core algorithm.
    match mode:
        case "sequential":
            return VO_sequential(fitness_f)
        case "threading":
            return V1_threading(fitness_f)
        case "multiprocessing":
            return V2_multiprocessing(fitness_f)
        case "async":
            return V3_async(fitness_f, latency_range=latency_range)
        case "vectorized":
            if vectorized_f is None:
                raise ValueError("Mode 'vectorized' requires a vectorized_f function.")
            return V4_vectorized(vectorized_f)
        case _:
            raise ValueError(f"Invalid mode: '{mode}'")