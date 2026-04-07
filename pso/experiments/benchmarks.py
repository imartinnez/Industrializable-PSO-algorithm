# @author: Íñigo Martínez Jiménez
# This module defines the benchmark utilities used to build and run PSO 
# experiments, including the Instance data container, a helper to generate 
# experiment configurations, and a function to execute a full benchmark suite


from dataclasses import dataclass
from typing import Callable

from pso.parallel.evaluator import choose_evaluator
import pso.core.pso_engine as p
import pso.objectives.registry as r


# We use @dataclass because automatically generates useful methods such as 
# __init__, __repr__, and this class is only meant to store data
@dataclass
class Instance:
    """
    Store the configuration of a single PSO experiment.

    Args:
        name (str): Name of the experiment instance.
        fitness_f (Callable): Objective function to optimize.
        dim (int): Dimension of the function.
        constraints (tuple[float, float]): Lower and upper bounds of the search space.
        seed (int): Random seed used for reproducibility.
        max_iter (int): Maximum number of iterations.
        n_particles (int): Number of particles in the swarm.
        strategy (str): Boundary handling strategy.
        fitness_policy (str): Fitness evaluation policy.
        mode (str): Evaluation mode used to compute fitness values.
        topology (str): Swarm topology.
        patience (float): Number of iterations allowed without meaningful improvement.
        imp_min (float): Minimum improvement required to reset the patience counter.
        tol (float): Tolerance used for early stopping near the optimum.
        w (float): Inertia coefficient.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        optimum_value (float | None): Known optimum value of the function, if available.
    """
    name: str
    fitness_f: Callable
    dim: int
    constraints: tuple[float, float]
    seed: int
    max_iter: int
    n_particles: int
    strategy: str
    fitness_policy: str
    mode: str
        
    topology: str = "global"
    patience: float = 100
    imp_min: float = 1e-8
    tol: float = 0.0
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    optimum_value: float | None = None

    def run_instance(self):
        """
        Run one PSO experiment instance with its stored configuration.

        Returns:
            p.r.Result: Result of the PSO execution.
        """""
        evaluator = choose_evaluator(self.mode, self.fitness_f)

        try:
            pso = p.PSO(
                n_particles=self.n_particles,
                fitness_f=self.fitness_f,
                dim=self.dim,
                constraints=self.constraints,
                strategy=self.strategy,
                fitness_policy=self.fitness_policy,
                topology=self.topology,
                tol=self.tol,
                max_iter=self.max_iter,
                patience = self.patience,
                imp_min = self.imp_min,
                evaluator=evaluator,
                w=self.w,
                c1=self.c1,
                c2=self.c2,
                optimum_value=self.optimum_value
            )

            result = pso.run(seed=self.seed)
        finally:
            evaluator.shutdown()   # ← libera el pool siempre, incluso si hay error
        
        return result




def make_instances(objectives: list[str], dims: list[int], seeds: list[int], max_iter: int, n_particles: int, strategy: str, fitness_policy: str, mode: str, 
                   topology: str = "global", patience: int = 100, imp_min: float = 1e-8, tol: float = 0.0, w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> list[Instance]:
    """
    Build a list of experiment instances from a set of objectives, dimensions, and seeds.

    Args:
        objectives (list[str]): Names of the objective functions to test.
        dims (list[int]): Dimensions to evaluate.
        seeds (list[int]): Random seeds to use.
        max_iter (int): Maximum number of iterations.
        n_particles (int): Number of particles in the swarm.
        strategy (str): Boundary handling strategy.
        fitness_policy (str): Fitness evaluation policy.
        mode (str): Evaluation mode used to compute fitness values.
        topology (str): Swarm topology.
        patience (int): Number of iterations allowed without meaningful improvement.
        imp_min (float): Minimum improvement required to reset the patience counter.
        tol (float): Tolerance used for early stopping near the optimum.
        w (float): Inertia coefficient.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.

    Returns:
        list[Instance]: List of configured experiment instances.
    """
    instances = []

    for objective_name in objectives:
        objective = r.get_objective(objective_name)

        for dim in dims:
            for seed in seeds:
                instances.append(
                    Instance(
                        name=f"{objective_name}_d{dim}_s{seed}",
                        fitness_f=objective.function,
                        dim=dim,
                        constraints=objective.constraints,
                        seed=seed,
                        max_iter=max_iter,
                        patience=patience,
                        imp_min=imp_min,
                        n_particles=n_particles,
                        strategy=strategy,
                        fitness_policy=fitness_policy,
                        mode=mode,
                        topology=topology,
                        tol=tol,
                        optimum_value=objective.optimum_value,
                        w=w,
                        c1=c1,
                        c2=c2
                    )
                )

    return instances


def run_suite(instances: list[Instance]) -> list[dict]:
    """
    Run a full benchmark suite and store the main result of each instance.

    Args:
        instances (list[Instance]): Experiment instances to execute.

    Returns:
        list[dict]: List of dictionaries with the most important result metrics.
    """
    results = []

    for instance in instances:
        result = instance.run_instance()

        row = {
            "instance_name": instance.name,
            "dim": instance.dim,
            "seed": instance.seed,
            "strategy": instance.strategy,
            "fitness_policy": instance.fitness_policy,
            "mode": instance.mode,
            "topology": instance.topology,
            "best_position": result.b_position,
            "best_value": result.b_value,
            "total_time": result.total_time,
            "fitness_eval_time_total": result.fitness_eval_time_total,
            "best_fitness_by_iter": result.best_fitness_by_iter,
            "iterations_executed": result.iterations,
        }

        results.append(row)

    return results