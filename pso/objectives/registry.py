# @author: Íñigo Martínez Jiménez

"""
This module defines the objective function registry used in the PSO,
storing the benchmark functions together with their search bounds,
known optimum values, and optimum points.
"""

from collections.abc import Callable
import numpy as np

import pso.objectives.functions as f


class Objective:
    """
    Store the main information associated with one benchmark objective function.
    """

    def __init__(self, name: str, function: Callable[[np.ndarray], float], constraints: tuple[float, float], 
                 optimum_value: float, optimum_point: Callable[[int], np.ndarray]) -> None:
        """
        Initialize an objective function with its metadata.

        Args:
            name (str): Name of the objective function.
            function (Callable[[np.ndarray], float]): Function to evaluate.
            constraints (tuple[float, float]): Lower and upper bounds of the search space.
            optimum_value (float): Known optimum value of the function.
            optimum_point (Callable[[int], np.ndarray]): Function that returns the optimum point for a given dimension.
        """
        self.name = name
        self.function = function
        self.constraints = constraints
        self.optimum_value = optimum_value
        self.optimum_point = optimum_point


OBJECTIVES = {
    "sphere": Objective(
        name="sphere",
        function=f.sphere,
        constraints=(-5.12, 5.12),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
    "rosenbrock": Objective(
        name="rosenbrock",
        function=f.rosenbrock,
        constraints=(-5.0, 10),
        optimum_value=0.0,
        optimum_point=lambda dim: np.ones(dim),
    ),
    "rastrigin": Objective(
        name="rastrigin",
        function=f.rastrigin,
        constraints=(-5.12, 5.12),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
    "ackley": Objective(
        name="ackley",
        function=f.ackley,
        constraints=(-32.768, 32.768),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
}


def get_objective(name: str) -> Objective:
    """
    Return the objective function associated with the given name.

    Args:
        name (str): Name of the objective function.

    Raises:
        ValueError: If the objective name is not registered.

    Returns:
        Objective: Registered objective object.
    """
    if name in OBJECTIVES:
        return OBJECTIVES[name]
    else:
        raise ValueError(f"Unknown function: {name}")


def optimum_point(objective: Objective, dim: int) -> np.ndarray:
    """
    Return the optimum point of an objective function for a given dimension.

    Args:
        objective (Objective): Objective function object.
        dim (int): Dimension of the function.

    Returns:
        np.ndarray: Optimum point for the given dimension.
    """
    return objective.optimum_point(dim)


def optimum_value(objective: Objective) -> float:
    """
    Return the optimum value of an objective function.

    Args:
        objective (Objective): Objective function object.

    Returns:
        float: Known optimum value.
    """
    return objective.optimum_value


def bounds_array(objective: Objective, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the lower and upper bound arrays for a given dimension.

    Args:
        objective (Objective): Objective function object.
        dim (int): Dimension of the function.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bound arrays.
    """
    lower = np.full(dim, objective.constraints[0])
    upper = np.full(dim, objective.constraints[1])
    return lower, upper