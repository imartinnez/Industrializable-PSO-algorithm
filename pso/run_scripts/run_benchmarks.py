from pathlib import Path
import json
import pandas as pd
from datetime import datetime

import pso.experiments.benchmarks as b
import pso.objectives.registry as r

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
    modes = ["sequential", "threading", "multiprocessing"]

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"benchmarks_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for mode in modes:
        instances = b.make_instances(
            objectives=objectives,
            dims=dims,
            seeds=seeds,
            mode=mode,
            **common
        )

        rows = b.run_suite(instances)
        rows = complete_results(rows)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "benchmark_results.csv", index=False)

    config = {
        "objectives": objectives,
        "dims": dims,
        "seeds": seeds,
        "modes": modes,
        **common,
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # A veces se crean varias carpetas a la vez
    if not any(outdir.iterdir()):
        outdir.rmdir()

    print(df.groupby(["objective", "dim", "mode"])["gap_to_optimum"].agg(["mean", "std"]))