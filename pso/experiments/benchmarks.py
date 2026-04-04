from dataclasses import dataclass
from typing import Callable, Tuple

from pso.parallel.evaluator import choose_evaluator
import pso.core.pso_engine as p
import pso.objectives.registry as r

@dataclass
class Instance:
    name: str
    fitness_f: Callable
    dim: int
    constraints: Tuple[float, float]
    seed: int
    max_iter: int
    n_particles: int
    strategy: str
    mode: str
        
    topology: str = "global"
    tol: float = 0.0
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5

    def run_instance(self):
        
        evaluator = choose_evaluator(self.mode, self.fitness_f)

        pso = p.PSO(
            n_particles=self.n_particles,
            fitness_f=self.fitness_f,
            dim=self.dim,
            constraints=self.constraints,
            strategy=self.strategy,
            topology=self.topology,
            tol=self.tol,
            max_iter=self.max_iter,
            evaluator=evaluator,
            w=self.w,
            c1=self.c1,
            c2=self.c2
        )

        result = pso.run(seed=self.seed)

        return result




def make_instances(objectives, dims, seeds, max_iter, n_particles, strategy, mode, topology="global", tol=0.0, w=0.7, c1=1.5, c2=1.5):
    instances = []

    for objective_name in objectives:
        objective = r.get_objective(objective_name)

        for dim in dims:
            for seed in seeds:
                instances.append(
                    Instance(
                        name=f"{objective_name}_d{dim}_s{seed}", fitness_f=objective.function,
                        dim=dim,
                        constraints=objective.constraints,
                        seed=seed,
                        max_iter=max_iter,
                        n_particles=n_particles,
                        strategy=strategy,
                        mode=mode,
                        topology=topology,
                        tol=tol,
                        w=w,
                        c1=c1,
                        c2=c2
                    )
                )

    return instances

def run_suite(instances):
    results = []

    for instance in instances:
        result = instance.run_instance()

        row = {
            "instance_name": instance.name,
            "dim": instance.dim,
            "seed": instance.seed,
            "strategy": instance.strategy,
            "mode": instance.mode,
            "topology": instance.topology,
            "best_position": result.best_position,
            "best_value": result.best_value,
            "total_time": result.total_time,
            "fitness_eval_time_total": result.fitness_eval_time_total,
            "iterations_executed": result.iterations_executed,
            "best_fitness_by_iter": result.best_fitness_by_iter,
        }

        results.append(row)

    return results