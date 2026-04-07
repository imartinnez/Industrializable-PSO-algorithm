import pandas as pd
import os
import platform

import pso.experiments.benchmarks as b
import pso.objectives.registry as r
from pso.experiments.pyswarm_reference import run_pyswarm
from pso.io.paths import make_run_dir
from pso.io.logging import setup_logging
from pso.io.save_results import save_csv, save_json

def complete_results(rows):
    expanded = []

    for row in rows:
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
    objectives = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    dims = [2, 10, 30]
    seeds = [1, 2, 3, 4, 5]
    modes = ["sequential", "threading"]

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

    outdir = make_run_dir("benchmarks")
    logger = setup_logging("pso.run_benchmarks", outdir / "run_benchmarks.log")

    logger.info("Starting benchmarks | objectives=%s | dims=%s | seeds=%s | modes=%s", objectives, dims, seeds, modes)

    all_rows = []

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

    df = pd.DataFrame(all_rows)

    results_path = outdir / "benchmark_results.csv"
    save_csv(df, results_path)

    config = {
        "objectives": objectives,
        "dims": dims,
        "seeds": seeds,
        "modes": modes + ["pyswarm"],
        **common,
        "hardware": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": platform.python_version(),
        }
    }
    save_json(config, outdir / "config.json")

    summary = df.groupby(["objective", "dim", "mode"])["gap_to_optimum"].agg(["mean", "std"]).reset_index()

    summary_path = outdir / "benchmark_summary.csv"
    save_csv(summary, summary_path)

    logger.info("Benchmark results saved to %s", results_path)
    logger.info("Benchmark summary saved to %s", summary_path)
    logger.info("Summary:\n%s", summary.to_string(index=False))