# @author: Íñigo Martínez Jiménez
# This module defines a simple script to run and compare individual PSO
# executions, showing the results of two custom configurations and the
# PySwarm reference implementation in a clear console summary

from pyswarm import pso
import numpy as np

import pso.objectives.registry as o
import pso.experiments.benchmarks as i
from pso.io.paths import make_run_dir
from pso.io.logging import setup_logging


if __name__ == "__main__":

    # Create the output folder and logger for this run
    outdir = make_run_dir("single_run")
    logger = setup_logging("pso.run", outdir / "run.log")
    
    # Load the benchmark function used in this example
    sphere = o.get_objective("sphere")

    # First custom PSO run using the sequential evaluator
    instance1 = i.Instance(
        name="sphere_d10",
        fitness_f=sphere.function,
        dim=3,
        constraints=sphere.constraints,
        seed=1,
        max_iter=1000,
        patience=100,
        imp_min=1e-8,
        n_particles=50,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        tol=1e-50,
        mode="sequential",
        optimum_value=sphere.optimum_value
    )

    # Second custom PSO run using the multiprocessing evaluator
    instance2 = i.Instance(
        name="sphere2_d10",
        fitness_f=sphere.function,
        dim=3,
        constraints=sphere.constraints,
        seed=1,
        max_iter=1000,
        patience=50,
        imp_min=1e-6,
        n_particles=50,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        tol=1e-50,
        mode="multiprocessing",
        optimum_value=sphere.optimum_value
    )
    
    # Run the first instance and print its main metrics
    result1 = instance1.run_instance()

    pct1 = 100 * result1.fitness_eval_time_total / result1.total_time
    pos1 = "  ".join(f"{x}" for x in result1.b_position)
    print(f"\n{instance1.name}")
    print(f"  best value   {result1.b_value:.6e}")
    print(f"  iterations   {result1.iterations} / {instance1.max_iter}")
    print(f"  total time   {result1.total_time:.3f} s")
    print(f"  eval time    {result1.fitness_eval_time_total:.3f} s  ({pct1:.1f}%)")
    print(f"  best pos     [{pos1}]")




    # Run the second instance and print its main metrics
    result2 = instance2.run_instance()

    pct2 = 100 * result2.fitness_eval_time_total / result2.total_time
    pos2 = "  ".join(f"{x}" for x in result2.b_position)
    print(f"\n{instance2.name}")
    print(f"  best value   {result2.b_value:.6e}")
    print(f"  iterations   {result2.iterations} / {instance2.max_iter}")
    print(f"  total time   {result2.total_time:.3f} s")
    print(f"  eval time    {result2.fitness_eval_time_total:.3f} s  ({pct2:.1f}%)")
    print(f"  best pos     [{pos2}]\n\n\n")




    # Run PySwarm as an external reference with comparable parameters
    lb = [sphere.constraints[0]] * instance1.dim
    ub = [sphere.constraints[1]] * instance1.dim

    np.random.seed(1)
    xopt, fopt = pso(
        sphere.function,
        lb,
        ub,
        swarmsize=50,
        omega=0.7,
        phip=1.5,
        phig=1.5,
        maxiter=2000,
        minfunc=1e-50,
        minstep=1e-50
    )

    pos_ref = "  ".join(f"{x}" for x in xopt)
    print(f"\npyswarm (reference)")
    print(f"  best value   {fopt:.6e}")
    print(f"  best pos     [{pos_ref}]")

    sep = "-" * 57
    print(f"\n{sep}")
    print(f"  {'instance':<18} {'best value':>12}   {'iters':>6}   {'time':>8}")
    print(sep)
    print(f"  {instance1.name:<18} {result1.b_value:>12.3e}   {result1.iterations:>6}   {result1.total_time:>6.3f} s")
    print(f"  {instance2.name:<18} {result2.b_value:>12.3e}   {result2.iterations:>6}   {result2.total_time:>6.3f} s")
    print(f"  {'pyswarm':<18} {fopt:>12.3e}   {'--':>6}   {'--':>8}")
    print(sep)