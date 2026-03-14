import numpy as np

import pso.objectives.functions as f

class Objective:
    def __init__(self, name, function, constraints, optimum_value, optimum_point):
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
    lower = np.full(dim, objective.constraints[0])
    upper = np.full(dim, objective.constraints[1])
    return lower, upper