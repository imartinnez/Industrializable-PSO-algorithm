# @author: Íñigo Martínez Jiménez
# This module defines the utility used to run the PySwarm baseline
# excuting the external PySwarm implementation with the same objective
# function and main parameters, so its results can be compared with the custom PSO

from time import perf_counter
import contextlib
import io as sysio

import numpy as np
from pyswarm import pso

import pso.objectives.registry as r


def run_pyswarm(objective_name: str, dim: int, seed: int, *, n_particles: int, w: float, c1: float, c2: float, max_iter: int, tol: float) -> dict:
    """
    Run the PySwarm baseline on a given objective function.

    Args:
        objective_name (str): Name of the objective function.
        dim (int): Dimension of the function.
        seed (int): Random seed used for reproducibility.
        n_particles (int): Number of particles in the swarm.
        w (float): Inertia coefficient.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance used as stopping criterion.

    Returns:
        dict: Dictionary with the main execution results.
    """
    objective = r.get_objective(objective_name)

    lb = [objective.constraints[0]] * dim
    ub = [objective.constraints[1]] * dim

    np.random.seed(seed)
    start = perf_counter()

    # Silence PySwarm internal output by redirecting it to a fake stream
    with contextlib.redirect_stdout(sysio.StringIO()):
        xopt, fopt = pso(
            objective.function,
            lb,
            ub,
            swarmsize=n_particles,
            omega=w,
            phip=c1,
            phig=c2,
            maxiter=max_iter,
            minfunc=tol,
            minstep=tol,
        )

    # Calculates the total time
    total_time = perf_counter() - start

    return {
        "objective": objective_name,
        "dim": dim,
        "seed": seed,
        "mode": "pyswarm",
        "best_value": float(fopt),
        "gap_to_optimum": abs(float(fopt) - objective.optimum_value),
        "total_time": total_time,
        "best_position": xopt.tolist(),
    }