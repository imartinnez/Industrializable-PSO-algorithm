# @author: Íñigo Martínez Jiménez
# This module defines the entry point that applies the PSO optimizer to the
# data center cooling use case. It builds the scenario, evaluates the baseline,
# runs the PSO, compares both solutions, persists CSV/JSON outputs and plots,
# and prints a clear console summary

import numpy as np
import pandas as pd

from pso.experiments.benchmarks import Instance
from pso.io.paths import make_run_dir
from pso.io.logging import setup_logging
from pso.io.save_results import save_csv, save_json
from pso.use_case.scenario import create_default_datacenter_scenario
from pso.use_case.encoding import phys_to_norm
from pso.use_case.objective import evaluate_solution, make_objective
from pso.use_case.datacenter_plots import (
    plot_temperature_heatmap,
    plot_energy_comparison,
    plot_energy_breakdown,
    plot_convergence,
    plot_variable_choice,
)


if __name__ == "__main__":

    # Create the output folder and logger for this run
    outdir = make_run_dir("datacenter")
    logger = setup_logging("pso.datacenter", outdir / "run.log")

    # Build the scenario with a fixed seed so the experiment is reproducible
    seed = 42
    scenario = create_default_datacenter_scenario(seed=seed)
    logger.info(
        "Scenario built | n_racks=%d, n_fans=%d, n_zones=%d, dim=%d, baseline_energy=%.4f",
        scenario.n_racks, scenario.n_fans, scenario.n_zones, scenario.dim,
        scenario.baseline_energy,
    )

    # Evaluate the baseline configuration
    x_baseline_norm = phys_to_norm(scenario.baseline_x_phys, scenario)
    baseline = evaluate_solution(x_baseline_norm, scenario)
    logger.info(
        "Baseline | fitness=%.4f, total_energy=%.4f, max_T=%.2f, unsafe=%d, hotspots=%d",
        baseline["fitness"], baseline["total_energy"], baseline["max_temperature"],
        baseline["unsafe_racks"], baseline["hotspots"],
    )

    # Build the PSO instance using the existing Instance dataclass
    # The objective is closed over the scenario so the engine only sees f(x) -> float
    objective = make_objective(scenario)
    instance = Instance(
        name="datacenter_case",
        fitness_f=objective,
        dim=scenario.dim,
        constraints=(0.0, 1.0),     # everything is normalised to [0, 1]
        seed=seed,
        max_iter=500,
        n_particles=40,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        patience=100,
        imp_min=1e-8,
        tol=0.0,                    # no known optimum
        w=0.7,
        c1=1.5,
        c2=1.5,
        optimum_value=None,
        mode="sequential",
    )

    logger.info(
        "Running PSO | dim=%d, particles=%d, max_iter=%d",
        instance.dim, instance.n_particles, instance.max_iter,
    )
    result = instance.run_instance()
    logger.info(
        "PSO finished | iterations=%d, total_time=%.3fs",
        result.iterations, result.total_time,
    )

    # Evaluate the PSO solution in detail
    pso_eval = evaluate_solution(result.b_position, scenario)
    energy_savings = (
        (baseline["total_energy"] - pso_eval["total_energy"]) / baseline["total_energy"] * 100
    )

    # Console summary
    print()
    print("=" * 70)
    print("DATA CENTER COOLING — PSO vs BASELINE")
    print("=" * 70)
    print(f"{'metric':<32} {'baseline':>15} {'PSO':>15}")
    print("-" * 70)
    print(f"{'fitness':<32} {baseline['fitness']:>15.4f} {pso_eval['fitness']:>15.4f}")
    print(f"{'total energy':<32} {baseline['total_energy']:>15.4f} {pso_eval['total_energy']:>15.4f}")
    print(f"{'chiller energy':<32} {baseline['chiller_energy']:>15.4f} {pso_eval['chiller_energy']:>15.4f}")
    print(f"{'fan energy':<32} {baseline['fan_energy']:>15.4f} {pso_eval['fan_energy']:>15.4f}")
    print(f"{'airflow energy':<32} {baseline['airflow_energy']:>15.4f} {pso_eval['airflow_energy']:>15.4f}")
    print(f"{'max temperature (C)':<32} {baseline['max_temperature']:>15.2f} {pso_eval['max_temperature']:>15.2f}")
    print(f"{'mean temperature (C)':<32} {baseline['mean_temperature']:>15.2f} {pso_eval['mean_temperature']:>15.2f}")
    print(f"{'min temperature (C)':<32} {baseline['min_temperature']:>15.2f} {pso_eval['min_temperature']:>15.2f}")
    print(f"{'std temperature (C)':<32} {baseline['std_temperature']:>15.2f} {pso_eval['std_temperature']:>15.2f}")
    print(f"{'unsafe racks':<32} {baseline['unsafe_racks']:>15d} {pso_eval['unsafe_racks']:>15d}")
    print(f"{'hotspots':<32} {baseline['hotspots']:>15d} {pso_eval['hotspots']:>15d}")
    print("-" * 70)
    print(f"Energy savings vs baseline:  {energy_savings:+.2f}%")
    print(f"PSO runtime:                 {result.total_time:.3f} s")
    print(f"PSO iterations executed:     {result.iterations}")
    print()
    print("Optimal configuration found by PSO:")
    print(f"  T_set    = {pso_eval['t_set']:.3f} C")
    print(f"  fans     = {np.array_str(pso_eval['fan_speeds'], precision=3)}")
    print(f"  airflows = {np.array_str(pso_eval['zone_airflows'], precision=3)}")
    print("=" * 70)
    print()

    # Persist results
    # Summary CSV with one row per source (baseline and PSO) and the main metrics
    flat_keys = [
        "fitness", "total_energy", "chiller_energy", "fan_energy", "airflow_energy",
        "mean_temperature", "max_temperature", "min_temperature", "std_temperature",
        "unsafe_racks", "hotspots", "overcooled_racks",
    ]
    summary_rows = [
        {"source": "baseline", **{k: baseline[k] for k in flat_keys}},
        {"source": "pso", **{k: pso_eval[k] for k in flat_keys}},
    ]
    save_csv(pd.DataFrame(summary_rows), outdir / "summary.csv")

    # Convergence curve as a CSV for later analysis
    convergence_df = pd.DataFrame({
        "iteration": np.arange(1, len(result.best_fitness_by_iter) + 1),
        "best_fitness": result.best_fitness_by_iter,
    })
    save_csv(convergence_df, outdir / "convergence.csv")

    # Full configuration in JSON so the experiment can be reproduced later
    save_json(
        {
            "seed": seed,
            "scenario": {
                "n_racks": scenario.n_racks,
                "n_fans": scenario.n_fans,
                "n_zones": scenario.n_zones,
                "grid_shape": list(scenario.grid_shape),
                "baseline_energy": scenario.baseline_energy,
            },
            "pso_config": {
                "dim": instance.dim,
                "n_particles": instance.n_particles,
                "max_iter": instance.max_iter,
                "patience": instance.patience,
                "w": instance.w, "c1": instance.c1, "c2": instance.c2,
                "strategy": instance.strategy,
                "topology": instance.topology,
                "mode": instance.mode,
            },
            "baseline_solution": {
                "fitness": baseline["fitness"],
                "total_energy": baseline["total_energy"],
                "max_temperature": baseline["max_temperature"],
                "mean_temperature": baseline["mean_temperature"],
                "unsafe_racks": baseline["unsafe_racks"],
                "hotspots": baseline["hotspots"],
            },
            "pso_solution": {
                "fitness": pso_eval["fitness"],
                "total_energy": pso_eval["total_energy"],
                "max_temperature": pso_eval["max_temperature"],
                "mean_temperature": pso_eval["mean_temperature"],
                "unsafe_racks": pso_eval["unsafe_racks"],
                "hotspots": pso_eval["hotspots"],
                "t_set": pso_eval["t_set"],
                "fan_speeds": pso_eval["fan_speeds"].tolist(),
                "zone_airflows": pso_eval["zone_airflows"].tolist(),
            },
            "energy_savings_pct": energy_savings,
            "pso_runtime_seconds": result.total_time,
            "pso_iterations": result.iterations,
        },
        outdir / "config.json",
    )

    # Plots
    plot_temperature_heatmap(baseline["temperatures"], scenario,
                              title="Baseline temperatures",
                              path=outdir / "heatmap_baseline.png")
    plot_temperature_heatmap(pso_eval["temperatures"], scenario,
                              title="PSO temperatures",
                              path=outdir / "heatmap_pso.png")
    plot_energy_comparison(baseline, pso_eval, outdir / "energy_comparison.png")
    plot_energy_breakdown(baseline, pso_eval, outdir / "energy_breakdown.png")
    plot_convergence(result.best_fitness_by_iter, outdir / "convergence.png")
    plot_variable_choice(scenario, baseline, pso_eval, outdir / "variable_choice.png")

    logger.info("Summary saved to %s", outdir / "summary.csv")
    logger.info("Convergence saved to %s", outdir / "convergence.csv")
    logger.info("Config saved to %s", outdir / "config.json")
    logger.info("Plots saved in %s", outdir)
