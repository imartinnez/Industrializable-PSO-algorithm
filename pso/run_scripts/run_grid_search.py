from itertools import product
from pathlib import Path
from datetime import datetime
import pandas as pd

import pso.objectives.registry as o
import pso.experiments.benchmarks as b


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
    objective_name,
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
):
    rows = []
    i = 1

    for dim, seed, mode, n_particles, w, c1, c2, max_iter in product(
        dims, seeds, modes, n_particles_grid, w_grid, c1_grid, c2_grid, max_iter_grid
    ):
        objective = o.get_objective(objective_name)

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
        )

        result = instance.run_instance()

        gap = abs(result.b_value - objective.optimum_value)
        auc_fitness = convergence_auc(result.best_fitness_by_iter)
        auc_gap = convergence_auc_gap(result.best_fitness_by_iter, objective.optimum_value)

        print(
            f"[{i}] obj={objective_name} | dim={dim} | seed={seed} | mode={mode} "
            f"| particles={n_particles} | w={w} | c1={c1} | c2={c2} | max_iter={max_iter} "
            f"| best={result.b_value:.6e} | gap={gap:.6e} | auc_gap={auc_gap:.6e} "
            f"| iter={result.iterations} | time={result.total_time:.6f}"
        )

        rows.append({
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
        })

        i += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    
    objective_names = ["sphere", "rosenbrock", "rastrigin", "ackley"]
    dims = [2, 10, 30]
    seeds = [1, 2, 3, 4, 5]
    modes = ["sequential", "threading"]

    n_particles_grid = [20, 30, 40, 50, 75, 100]
    w_grid = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c1_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    c2_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    max_iter_grid = [500, 1000, 2000]

    df = pso_grid_search(
        objective_name=objective_names,
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
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results")
    outdir.mkdir(exist_ok=True)

    raw_path = outdir / f"grid_search_full_{timestamp}.csv"
    df.to_csv(raw_path, index=False)

    print(f"\nRaw results saved in: {raw_path}")

    summary = (
        df.groupby(
            [
                "objective",
                "dim",
                "mode",
                "n_particles",
                "w",
                "c1",
                "c2",
                "max_iter",
            ],
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
        .sort_values(["objective", "dim", "mode", "mean_gap", "mean_auc_gap", "mean_time"])
    )

    summary_path = outdir / f"grid_search_full_{timestamp}_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = (
        summary.groupby(["objective", "dim", "mode"], as_index=False)
        .first()
    )

    best_path = outdir / f"grid_search_full_{timestamp}_best.csv"
    best.to_csv(best_path, index=False)

    print(f"Summary saved in: {summary_path}")
    print(f"Best configs saved in: {best_path}")

    print("\nBest configuration by objective / dim / mode")
    print(best)