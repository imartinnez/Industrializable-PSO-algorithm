# @author: Íñigo Martínez Jiménez
# This module defines the plotting helpers used by the data center cooling
# use case. They are kept separate from pso/viz/, which is reserved for the
# swarm animations of the benchmark functions

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pso.use_case.scenario import DataCenterScenario


def plot_temperature_heatmap(temperatures: np.ndarray, scenario: DataCenterScenario,
                              title: str, path: Path) -> None:
    """
    Plot a heatmap of rack temperatures using the grid layout of the scenario.

    Args:
        temperatures (np.ndarray): Rack temperatures of shape (n_racks,).
        scenario (DataCenterScenario): Scenario holding the grid layout.
        title (str): Title of the figure.
        path (Path): Output path for the saved figure.
    """
    rows, cols = scenario.grid_shape
    grid = np.zeros((rows, cols))
    for r, (i, j) in enumerate(scenario.rack_positions.astype(int)):
        grid[i, j] = temperatures[r]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="inferno", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    # Overlay the actual temperature on each rack cell
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=9)

    fig.colorbar(im, ax=ax, label="T (°C)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_energy_comparison(baseline: dict, pso: dict, path: Path) -> None:
    """
    Plot a simple bar chart comparing the total energy of baseline and PSO.

    Args:
        baseline (dict): Baseline evaluation result.
        pso (dict): PSO evaluation result.
        path (Path): Output path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["baseline", "PSO"]
    values = [baseline["total_energy"], pso["total_energy"]]
    bars = ax.bar(labels, values, color=["gray", "tab:blue"])

    ax.set_ylabel("total energy")
    ax.set_title("Total energy: baseline vs PSO")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_energy_breakdown(baseline: dict, pso: dict, path: Path) -> None:
    """
    Plot a grouped bar chart of the chiller, fans, and airflow energy components.

    Args:
        baseline (dict): Baseline evaluation result.
        pso (dict): PSO evaluation result.
        path (Path): Output path for the saved figure.
    """
    components = ["chiller_energy", "fan_energy", "airflow_energy"]
    labels = ["chiller", "fans", "airflow"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, [baseline[c] for c in components], width,
           label="baseline", color="gray")
    ax.bar(x + width / 2, [pso[c] for c in components], width,
           label="PSO", color="tab:blue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("energy")
    ax.set_title("Energy breakdown")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(best_fitness_by_iter: list[float], path: Path) -> None:
    """
    Plot the PSO convergence curve.

    Args:
        best_fitness_by_iter (list[float]): Best fitness history of the run.
        path (Path): Output path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, len(best_fitness_by_iter) + 1)
    ax.plot(x, best_fitness_by_iter, color="tab:blue")

    ax.set_xlabel("iteration")
    ax.set_ylabel("best fitness")
    ax.set_title("PSO convergence")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_variable_choice(scenario: DataCenterScenario, baseline: dict, pso: dict,
                          path: Path) -> None:
    """
    Plot a grouped bar chart comparing baseline and PSO fan speeds and zone airflows.

    Args:
        scenario (DataCenterScenario): Scenario holding the baseline configuration.
        baseline (dict): Baseline evaluation result.
        pso (dict): PSO evaluation result.
        path (Path): Output path for the saved figure.
    """
    baseline_fans = scenario.baseline_x_phys[1:1 + scenario.n_fans]
    baseline_zones = scenario.baseline_x_phys[1 + scenario.n_fans:]

    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x_fans = np.arange(scenario.n_fans)
    axes[0].bar(x_fans - width / 2, baseline_fans, width, label="baseline", color="gray")
    axes[0].bar(x_fans + width / 2, pso["fan_speeds"], width, label="PSO", color="tab:blue")
    axes[0].set_xticks(x_fans)
    axes[0].set_xticklabels([f"fan {i}" for i in range(scenario.n_fans)])
    axes[0].set_ylabel("fan speed")
    axes[0].set_title("Fan speeds")
    axes[0].legend()

    x_zones = np.arange(scenario.n_zones)
    axes[1].bar(x_zones - width / 2, baseline_zones, width, label="baseline", color="gray")
    axes[1].bar(x_zones + width / 2, pso["zone_airflows"], width, label="PSO", color="tab:blue")
    axes[1].set_xticks(x_zones)
    axes[1].set_xticklabels([f"zone {i}" for i in range(scenario.n_zones)])
    axes[1].set_ylabel("zone airflow")
    axes[1].set_title("Zone airflows")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
