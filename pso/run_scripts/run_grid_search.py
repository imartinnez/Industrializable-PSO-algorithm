from itertools import product
import pandas as pd

import pso.objectives.registry as o
import pso.experiments.benchmarks as b
from pso.experiments.pyswarm_reference import run_pyswarm
from pso.io.paths import make_run_dir
from pso.io.logging import setup_logging
from pso.io.save_results import save_csv, save_json


def convergence_auc(curve):
    """AUC simple de la curva de convergencia."""
    if len(curve) == 0:
        return float("inf")
    return sum(curve) / len(curve)


def convergence_auc_gap(curve, optimum_value):
    """AUC del gap al óptimo por iteración."""
    if len(curve) == 0:
        return float("inf")
    gaps = [abs(v - optimum_value) for v in curve]
    return sum(gaps) / len(gaps)


def pso_grid_search(
    objective_names,
    dims,
    seeds,
    modes,
    n_particles_grid,
    w_grid,
    c1_grid,
    c2_grid,
    max_iter_grid,
    strategy="clamp",
    fitness_policy="plain",
    topology="global",
    patience=100,
    imp_min=1e-8,
    tol=1e-12,
    logger=None,
):
    rows = []
    i = 1

    for objective_name in objective_names:
        objective = o.get_objective(objective_name)

        for dim, seed, mode, n_particles, w, c1, c2, max_iter in product(
            dims, seeds, modes, n_particles_grid, w_grid, c1_grid, c2_grid, max_iter_grid
        ):
            if mode == "pyswarm":
                result_row = run_pyswarm(
                    objective_name,
                    dim,
                    seed,
                    n_particles=n_particles,
                    w=w,
                    c1=c1,
                    c2=c2,
                    max_iter=max_iter,
                    tol=tol,
                )

                row = {
                    "objective": objective_name,
                    "dim": dim,
                    "seed": seed,
                    "mode": "pyswarm",
                    "n_particles": n_particles,
                    "w": w,
                    "c1": c1,
                    "c2": c2,
                    "max_iter": max_iter,
                    "strategy": "internal",
                    "fitness_policy": "plain",
                    "topology": "global",
                    "patience": None,
                    "imp_min": None,
                    "tol": tol,
                    "best_value": result_row["best_value"],
                    "gap_to_optimum": result_row["gap_to_optimum"],
                    "auc_fitness": None,
                    "auc_gap": None,
                    "total_time": result_row["total_time"],
                    "fitness_eval_time_total": None,
                    "iterations": None,
                    "best_position": result_row["best_position"],
                }
            else:
                instance = b.Instance(
                    name=f"{objective_name}_d{dim}_s{seed}_{mode}",
                    fitness_f=objective.function,
                    dim=dim,
                    constraints=objective.constraints,
                    seed=seed,
                    max_iter=max_iter,
                    n_particles=n_particles,
                    strategy=strategy,
                    fitness_policy=fitness_policy,
                    mode=mode,
                    topology=topology,
                    patience=patience,
                    imp_min=imp_min,
                    tol=tol,
                    w=w,
                    c1=c1,
                    c2=c2,
                    optimum_value=objective.optimum_value,
                )

                result = instance.run_instance()

                gap = abs(result.b_value - objective.optimum_value)
                auc_fitness = convergence_auc(result.best_fitness_by_iter)
                auc_gap = convergence_auc_gap(result.best_fitness_by_iter, objective.optimum_value)

                row = {
                    "objective": objective_name,
                    "dim": dim,
                    "seed": seed,
                    "mode": mode,
                    "n_particles": n_particles,
                    "w": w,
                    "c1": c1,
                    "c2": c2,
                    "max_iter": max_iter,
                    "strategy": strategy,
                    "fitness_policy": fitness_policy,
                    "topology": topology,
                    "patience": patience,
                    "imp_min": imp_min,
                    "tol": tol,
                    "best_value": result.b_value,
                    "gap_to_optimum": gap,
                    "auc_fitness": auc_fitness,
                    "auc_gap": auc_gap,
                    "total_time": result.total_time,
                    "fitness_eval_time_total": result.fitness_eval_time_total,
                    "iterations": result.iterations,
                    "best_position": result.b_position.tolist() if hasattr(result.b_position, "tolist") else result.b_position,
                }

            rows.append(row)

            if logger is not None:
                logger.info(
                    "[%s] obj=%s | dim=%s | seed=%s | mode=%s | particles=%s | w=%.2f | c1=%.2f | c2=%.2f | max_iter=%s | best=%.6e | gap=%.6e | time=%.6f",
                    i,
                    row["objective"],
                    row["dim"],
                    row["seed"],
                    row["mode"],
                    row["n_particles"],
                    row["w"],
                    row["c1"],
                    row["c2"],
                    row["max_iter"],
                    row["best_value"],
                    row["gap_to_optimum"],
                    row["total_time"],
                )

            i += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    objective_names = ["sphere", "rastrigin"]
    dims = [2, 10]
    seeds = [1, 2]
    modes = ["sequential", "threading", "pyswarm"]

    n_particles_grid = [30, 50]
    w_grid = [0.5, 0.7]
    c1_grid = [1.0, 1.5]
    c2_grid = [1.0, 1.5]
    max_iter_grid = [500]

    outdir = make_run_dir("grid_search")
    logger = setup_logging("pso.run_grid_search", outdir / "run_grid_search.log")

    logger.info("Starting grid search")

    df = pso_grid_search(
        objective_names=objective_names,
        dims=dims,
        seeds=seeds,
        modes=modes,
        n_particles_grid=n_particles_grid,
        w_grid=w_grid,
        c1_grid=c1_grid,
        c2_grid=c2_grid,
        max_iter_grid=max_iter_grid,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        patience=100,
        imp_min=1e-8,
        tol=1e-12,
        logger=logger,
    )

    raw_path = outdir / "grid_search_full.csv"
    save_csv(df, raw_path)

    config = {
        "objective_names": objective_names,
        "dims": dims,
        "seeds": seeds,
        "modes": modes,
        "n_particles_grid": n_particles_grid,
        "w_grid": w_grid,
        "c1_grid": c1_grid,
        "c2_grid": c2_grid,
        "max_iter_grid": max_iter_grid,
        "strategy": "clamp",
        "fitness_policy": "plain",
        "topology": "global",
        "patience": 100,
        "imp_min": 1e-8,
        "tol": 1e-12,
    }
    save_json(config, outdir / "config.json")

    summary = (
        df.groupby(
            ["objective", "dim", "mode", "n_particles", "w", "c1", "c2", "max_iter"],
            as_index=False,
        )
        .agg(
            mean_best_value=("best_value", "mean"),
            std_best_value=("best_value", "std"),
            mean_gap=("gap_to_optimum", "mean"),
            std_gap=("gap_to_optimum", "std"),
            mean_auc_fitness=("auc_fitness", "mean"),
            mean_auc_gap=("auc_gap", "mean"),
            mean_time=("total_time", "mean"),
            std_time=("total_time", "std"),
            mean_fitness_eval_time=("fitness_eval_time_total", "mean"),
            mean_iterations=("iterations", "mean"),
        )
        .sort_values(["objective", "dim", "mode", "mean_gap", "mean_time"])
    )

    summary_path = outdir / "grid_search_summary.csv"
    save_csv(summary, summary_path)

    best = summary.groupby(["objective", "dim", "mode"], as_index=False).first()
    best_path = outdir / "grid_search_best.csv"
    save_csv(best, best_path)

    logger.info("Raw results saved to %s", raw_path)
    logger.info("Summary saved to %s", summary_path)
    logger.info("Best configs saved to %s", best_path)
    logger.info("Best configuration by objective / dim / mode:\n%s", best.to_string(index=False))