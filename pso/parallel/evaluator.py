# @author: Íñigo Martínez Jiménez

"""
This module defines the evaluator selector used in the PSO.
It centralizes the logic for choosing how particle fitness is evaluated, so the
PSO engine does not need to be modified every time a new evaluator is added.
"""

from typing import Any

from pso.parallel.V0_sequential import VO_sequential
from pso.parallel.V1_threading import V1_threading
from pso.parallel.V2_multiprocessing import V2_multiprocessing


def choose_evaluator(mode: str, fitness_f) -> Any:
    """
    Return the evaluator associated with the selected execution mode.

    Args:
        mode (str): Evaluation mode to use.
        fitness_f: Objective function to evaluate.

    Raises:
        ValueError: If the selected mode is not valid.

    Returns:
        Any: Evaluator object used to compute fitness values.
    """
    # This function isolates the evaluator selection logic from the PSO engine.
    # That way, new evaluators can be added without changing the core algorithm.
    match mode:
        case "sequential":
            return VO_sequential(fitness_f)
        case "threading":
            return V1_threading(fitness_f)
        case "multiprocessing":
            return V2_multiprocessing(fitness_f)  # Chunksize could also be added here as a parameter
        case "async":
            return
        case "vectorized":
            return
        case _:
            raise ValueError("Invalid mode")