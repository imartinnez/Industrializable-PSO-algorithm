import numpy as np

from .functions import *

class Objective:
    def __init__(self, name, function, lower_bound, upper_bound, optimum_value, optimum_point):
        self.name = name
        self.function = function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.optimum_value = optimum_value
        self.optimum_point = optimum_point
        
        
OBJECTIVES = {
    "sphere": Objective(
        name="sphere",
        function=sphere,
        constraints=(-5.12, 5.12),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
    "rosenbrock": Objective(
        name="rosenbrock",
        function=rosenbrock,
        constraints=(-5.0, 10),
        optimum_value=0.0,
        optimum_point=lambda dim: np.ones(dim),
    ),
    "rastrigin": Objective(
        name="rastrigin",
        function=rastrigin,
        constraints=(-5.12, 5.12),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
    "ackley": Objective(
        name="ackley",
        function=ackley,
        constraints=(-32.768, 32.768),
        optimum_value=0.0,
        optimum_point=lambda dim: np.zeros(dim),
    ),
}

def get_objective(name):
    if name in OBJECTIVES:
        return OBJECTIVES[name]
    else:
        raise ValueError(f"Unknown function: {name}")

def optimum_point(objective, dim):
    #Genera el punto óptimo de una función para una dimensión dada
    return objective.optimum_point(dim)

def bounds_array(objective, dim):
    #Genera arrays con límites inferiores y superiores
    lower = np.full(dim, objective.lower_bound)
    upper = np.full(dim, objective.upper_bound)
    return lower, upper