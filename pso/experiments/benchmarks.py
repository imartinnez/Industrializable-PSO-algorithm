from dataclasses import dataclass
from typing import Callable, Tuple

import pso.core.pso as p

@dataclass
class Instance:
    name: str
    objective: Callable
    dim: int
    constraints: Tuple[float, float]
    seed: int
    max_iter: int
    n_particles: int
    strategy: str
    topology: str = "global"
    tol: float = 0.0
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5

    def run_instance(self):

        pso = p.PSO(
            n_particles=self.n_particles,
            fitness_f=self.objective,
            dim=self.dim,
            constraints=self.constraints,
            strategy=self.strategy,
            topology=self.topology,
            tol=self.tol,
            max_iter=self.max_iter,
            w=self.w,
            c1=self.c1,
            c2=self.c2
        )

        result = pso.run(seed=self.seed)

        return result


