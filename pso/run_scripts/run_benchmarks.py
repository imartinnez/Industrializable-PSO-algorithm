# @author: Íñigo Martínez Jiménez
# This module defines the main benchmark script used to run PSO experiments,
# compare the custom implementation against the PySwarm baseline, and save
# the resulting metrics, configuration, and summary files for later analysis

import os
import platform
import pandas as pd

import pso.experiments.benchmarks as b
import pso.objectives.registry as r
from pso.experiments.pyswarm_reference import run_pyswarm
from pso.io.logging import setup_logging
from pso.io.paths import make_run_dir
from pso.io.save_results import save_csv, save_json


def complete_results(rows: list[dict]) -> list[dict]:
    """
    Add extra benchmark metrics to the raw result rows.

    Args:
        rows (list[dict]): Raw result rows returned by the benchmark suite.

    Returns:
        list[dict]: Result rows enriched with the objective name and the gap to the optimum.
    """
    expanded = []

    for row in rows:
        # The objective name is extracted from the instance name
        objective_name = row["instance_name"].split("_d")[0]
        objective = r.get_objective(objective_name)
        optimum = objective.optimum_value

        expanded.append({
            "instance_name": row["instance_name"],
            "objective": objective_name,
            "dim": row["dim"],
            "seed": row["seed"],
            "mode": row["mode"],
            "strategy": row["strategy"],
            "fitness_policy": row["fitness_policy"],
            "topology": row["topology"],
            "best_value": row["best_value"],
            "gap_to_optimum": abs(row["best_value"] - optimum),
            "total_time": row["total_time"],
            "fitness_eval_time_total": row["fitness_eval_time_total"],
            "iterations_executed": row["iterations_executed"],
            "best_fitness_by_iter": row["best_fitness_by_iter"],
        })

    return expanded


if __name__ == "__main__":
    # Benchmark configuration
    objectives = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    dims = [2, 10, 30]
    seeds = [1, 2, 3, 4, 5]
    modes = ["sequential", "threading"]

    # Parameters shared by all custom PSO runs
    common = dict(
        max_iter=2000,
        n_particles=50,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        patience=100,
        imp_min=1e-8,
        tol=1e-12,
        w=0.7,
        c1=1.5,
        c2=1.5,
    )

    # Create the output folder and logger for this benchmark run
    outdir = make_run_dir("benchmarks")
    logger = setup_logging("pso.run_benchmarks", outdir / "run_benchmarks.log")

    logger.info("Starting benchmarks | objectives=%s | dims=%s | seeds=%s | modes=%s", objectives, dims, seeds, modes)

    all_rows = []

    # Run the custom PSO with each selected evaluation mode
    for mode in modes:
        logger.info("Running custom PSO mode=%s", mode)

        instances = b.make_instances(
            objectives=objectives,
            dims=dims,
            seeds=seeds,
            mode=mode,
            **common,
        )

        rows = b.run_suite(instances)
        rows = complete_results(rows)
        all_rows.extend(rows)

    logger.info("Running PySwarm baseline")

    # Run the reference PySwarm implementation with the same benchmark setup
    for objective_name in objectives:
        for dim in dims:
            for seed in seeds:
                row = run_pyswarm(
                    objective_name,
                    dim,
                    seed,
                    n_particles=common["n_particles"],
                    w=common["w"],
                    c1=common["c1"],
                    c2=common["c2"],
                    max_iter=common["max_iter"],
                    tol=common["tol"],
                )

                # Add the fields needed so PySwarm results follow the same schema
                # as the custom PSO benchmark results
                row.update({
                    "instance_name": f"{objective_name}_d{dim}_s{seed}_pyswarm",
                    "strategy": "internal",
                    "fitness_policy": "plain",
                    "topology": "global",
                    "fitness_eval_time_total": None,
                    "iterations_executed": None,
                    "best_fitness_by_iter": None,
                })

                all_rows.append(row)

    # Store the full benchmark results in a DataFrame
    df = pd.DataFrame(all_rows)

    results_path = outdir / "benchmark_results.csv"
    save_csv(df, results_path)

    # Save the benchmark configuration together with basic hardware information
    # so the experiment can be reproduced more easily later
    config = {
        "objectives": objectives,
        "dims": dims,
        "seeds": seeds,
        "modes": modes + ["pyswarm"],
        **common, # unpacks all key–value pairs from the common dictionary into config
        "hardware": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "cpu_count_logical": os.cpu_count(),
            "python_version": platform.python_version(),
        }
    }
    save_json(config, outdir / "config.json")

    # Build a compact summary using the gap to optimum
    # This makes it easier to compare performance across objectives, dimensions, and modes
    summary = df.groupby(["objective", "dim", "mode"])["gap_to_optimum"].agg(["mean", "std"]).reset_index()

    summary_path = outdir / "benchmark_summary.csv"
    save_csv(summary, summary_path)

    logger.info("Benchmark results saved to %s", results_path)
    logger.info("Benchmark summary saved to %s", summary_path)
    logger.info("Summary:\n%s", summary.to_string(index=False))