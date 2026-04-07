# @author: Íñigo Martínez Jiménez
# This module defines the visualization utilities used to generate detailed
# plots and animations of the PSO behaviour in 2D and 3D, including the swarm
# movement, the global best position, and the convergence curve over time.

from collections.abc import Callable
from pathlib import Path
from typing import Any
import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


logger = logging.getLogger(__name__)


def _build_grid(fitness_f: Callable[[np.ndarray], float], constraints: tuple[float, float], 
                resolution: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D grid of points and evaluate the objective function on it.

    Args:
        fitness_f (Callable[[np.ndarray], float]): Objective function to evaluate.
        constraints (tuple[float, float]): Lower and upper bounds of the search space.
        resolution (int): Number of points used per axis.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Grid coordinates and function values.
    """
    low, high = constraints
    xs = np.linspace(low, high, resolution)
    ys = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(xs, ys)

    # The objective function is scalar, so we evaluate it point by point
    # over the full grid in order to build the contour or surface background
    Z = np.vectorize(lambda xi, yi: fitness_f(np.array([xi, yi])))(X, Y)
    return X, Y, Z


def _fitness_frames(result: Any) -> list[float]:
    """
    Align the best fitness history with the number of stored animation frames.

    Args:
        result (Any): PSO result object containing trajectories and fitness history.

    Returns:
        list[float]: Fitness values matched to the saved frames.
    """
    # The optimizer may run more iterations than the number of stored frames,
    # so this function maps the full fitness history to the recorded animation steps
    n_frames = len(result.trajectories)
    n_full = result.iterations
    step = max(1, n_full // n_frames)

    return [
        result.best_fitness_by_iter[min(i * step, n_full - 1)]
        for i in range(n_frames)
    ]


def _save_or_show(anim: animation.FuncAnimation, save_path: str | Path | None, fps: int, dpi: int) -> None:
    """
    Save the animation to disk or display it interactively.

    Args:
        anim (animation.FuncAnimation): Animation object to save or display.
        save_path (str | Path | None): Output path. If None, the animation is shown instead.
        fps (int): Frames per second.
        dpi (int): Resolution used when saving video formats.
    """
    if save_path is None:
        plt.show()
        return

    save_path = Path(save_path)

    if save_path.suffix == ".gif":
        anim.save(str(save_path), writer="pillow", fps=fps)
    else:
        anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=dpi)

    logger.info("Saved animation to %s", save_path)


def animate_2d(result: Any, fitness_f: Callable[[np.ndarray], float], constraints: tuple[float, float], 
               title: str = "PSO 2D", save_path: str | Path | None = None, fps: int = 15, dpi: int = 120) -> animation.FuncAnimation:
    """
    Generate a detailed 2D animation of the PSO execution.

    The animation shows the swarm moving over the contour plot of the objective
    function, together with the evolution of the global best fitness.

    Args:
        result (Any): PSO result object containing trajectories and best positions.
        fitness_f (Callable[[np.ndarray], float]): Objective function to evaluate.
        constraints (tuple[float, float]): Lower and upper bounds of the search space.
        title (str): Title of the figure.
        save_path (str | Path | None): Output path for the animation. If None, it is shown on screen.
        fps (int): Frames per second.
        dpi (int): Resolution used when saving.

    Raises:
        ValueError: If the result object does not contain stored trajectories.

    Returns:
        animation.FuncAnimation: Generated animation object.
    """
    if result.trajectories is None:
        raise ValueError("Run PSO with stored trajectories before calling the animator.")

    trajs = result.trajectories
    gbests = result.best_positions_by_iter
    fit_frames = _fitness_frames(result)
    n_frames = len(trajs)
    low, high = constraints

    # Build the background contour of the objective function
    logger.info("Building 2D grid")
    X, Y, Z = _build_grid(fitness_f, constraints)

    # The figure is divided into two panels:
    # left -> swarm movement in the search space
    # right -> convergence of the global best fitness
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = GridSpec(1, 2, width_ratios=[3, 2], wspace=0.35)

    ax_s = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    # Create the contour background
    # A log scale is used when possible so the landscape is easier to read
    zmin = max(Z.min(), 1e-10)
    levels = np.exp(np.linspace(np.log(zmin), np.log(Z.max()), 40))
    ax_s.contourf(X, Y, Z, levels=levels, cmap="inferno", alpha=0.85)
    ax_s.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.2, alpha=0.3)

    ax_s.set_xlim(low, high)
    ax_s.set_ylim(low, high)
    ax_s.set_xlabel("x₁")
    ax_s.set_ylabel("x₂")

    # These are the plot objects that will be updated frame by frame
    scat_particles = ax_s.scatter([], [], s=20, c="cyan", edgecolors="white", linewidths=0.3, zorder=5, label="Particles")
    scat_gbest = ax_s.scatter([], [], s=180, c="red", marker="*", edgecolors="white", linewidths=0.4, zorder=6, label="Global best")
    ax_s.legend(loc="upper right", fontsize=8)

    # Small text box showing the current frame and the best fitness value
    txt = ax_s.text(0.02, 0.97, "", transform=ax_s.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    # Convergence panel
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Best fitness")
    ax_c.set_title("Convergence")
    ax_c.grid(True, alpha=0.3)

    # The full curve is shown in the background as a light reference,
    # while the animated line grows frame by frame
    ax_c.plot(range(n_frames), fit_frames, color="lightgray", linewidth=0.4, alpha=0.3)
    line_live, = ax_c.plot([], [], color="royalblue", linewidth=1.8)
    dot_live, = ax_c.plot([], [], "ro", markersize=6)

    ax_c.set_xlim(0, n_frames)
    pos_vals = [v for v in fit_frames if v > 0]
    if pos_vals:
        ax_c.set_ylim(min(pos_vals) * 0.1, max(fit_frames) * 10)
        ax_c.set_yscale("log")

    def init():
        """
        Initialize the animated artists with empty values.

        Returns:
            tuple: Artists that will be updated by FuncAnimation.
        """
        scat_particles.set_offsets(np.empty((0, 2)))
        scat_gbest.set_offsets(np.empty((0, 2)))
        line_live.set_data([], [])
        dot_live.set_data([], [])
        txt.set_text("")
        return scat_particles, scat_gbest, line_live, dot_live, txt

    def update(frame: int):
        """
        Update the 2D animation for one frame.

        Args:
            frame (int): Current frame index.

        Returns:
            tuple: Updated artists.
        """
        # Particle positions in the first two dimensions
        pos = trajs[frame][:, :2]
        scat_particles.set_offsets(pos)

        # Current global best position
        gp = gbests[frame][:2]
        scat_gbest.set_offsets([[gp[0], gp[1]]])

        # Update the convergence curve up to the current frame
        xs = list(range(frame + 1))
        ys = fit_frames[:frame + 1]
        line_live.set_data(xs, ys)
        dot_live.set_data([frame], [fit_frames[frame]])

        txt.set_text(f"Frame {frame+1}/{n_frames}\nBest: {fit_frames[frame]:.3e}")
        return scat_particles, scat_gbest, line_live, dot_live, txt

    # blit=True works well in 2D and makes the animation smoother
    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=1000 // fps, blit=True)

    _save_or_show(anim, save_path, fps, dpi)
    plt.close(fig)
    return anim


def animate_3d(result: Any, fitness_f: Callable[[np.ndarray], float], constraints: tuple[float, float], 
               title: str = "PSO 3D", save_path: str | Path | None = None, fps: int = 12, dpi: int = 110) -> animation.FuncAnimation:
    """
    Generate a detailed 3D animation of the PSO execution.

    The animation shows the swarm moving over the 3D surface of the objective
    function, together with the evolution of the global best fitness.

    Args:
        result (Any): PSO result object containing trajectories and best positions.
        fitness_f (Callable[[np.ndarray], float]): Objective function to evaluate.
        constraints (tuple[float, float]): Lower and upper bounds of the search space.
        title (str): Title of the figure.
        save_path (str | Path | None): Output path for the animation. If None, it is shown on screen.
        fps (int): Frames per second.
        dpi (int): Resolution used when saving.

    Raises:
        ValueError: If the result object does not contain stored trajectories.

    Returns:
        animation.FuncAnimation: Generated animation object.
    """
    if result.trajectories is None:
        raise ValueError("Run PSO with stored trajectories before calling the animator.")

    trajs = result.trajectories
    gbests = result.best_positions_by_iter
    fit_frames = _fitness_frames(result)
    n_frames = len(trajs)
    low, high = constraints

    logger.info("Building 3D grid")
    X, Y, Z = _build_grid(fitness_f, constraints, resolution=60)

    # Again, the figure is split into two panels:
    # left -> 3D swarm animation
    # right -> convergence history
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax_c = fig.add_subplot(gs[1])

    # Static 3D surface of the objective function
    ax3.plot_surface(X, Y, Z, cmap="inferno", alpha=0.5, linewidth=0, antialiased=True)
    ax3.set_xlabel("x₁", fontsize=8)
    ax3.set_ylabel("x₂", fontsize=8)
    ax3.set_zlabel("f(x)", fontsize=8)
    ax3.view_init(elev=28, azim=-60)

    # Computing z values inside every animation step would be too slow,
    # so we precompute them once before the animation starts
    logger.info("Precomputing particle fitness values")
    all_z = [np.array([fitness_f(p) for p in trajs[i]]) for i in range(n_frames)]
    all_gz = [fitness_f(gbests[i]) for i in range(n_frames)]

    # Initial scatter objects
    pos0 = trajs[0]
    sc_part = ax3.scatter(pos0[:, 0], pos0[:, 1], all_z[0], s=15, c="cyan", alpha=0.7, depthshade=True)
    gp0 = gbests[0]
    sc_gbest = ax3.scatter([gp0[0]], [gp0[1]], [all_gz[0]], s=140, c="red", marker="*", zorder=6)

    txt3 = ax3.text2D(0.02, 0.95, "", transform=ax3.transAxes, fontsize=8, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    # Convergence panel
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Best fitness")
    ax_c.set_title("Convergence")
    ax_c.grid(True, alpha=0.3)
    ax_c.plot(range(n_frames), fit_frames, color="lightgray", linewidth=0.4, alpha=0.3)
    line_live, = ax_c.plot([], [], color="royalblue", linewidth=1.8)
    dot_live, = ax_c.plot([], [], "ro", markersize=6)
    ax_c.set_xlim(0, n_frames)

    pos_vals = [v for v in fit_frames if v > 0]
    if pos_vals:
        ax_c.set_ylim(min(pos_vals) * 0.1, max(fit_frames) * 10)
        ax_c.set_yscale("log")

    # In 3D, blitting is not supported, so the full frame must be redrawn
    # A smooth camera rotation is added to better show the surface depth
    azims = np.linspace(-60, 300, n_frames)

    def update(frame: int):
        """
        Update the 3D animation for one frame.

        Args:
            frame (int): Current frame index.

        Returns:
            tuple: Updated artists.
        """
        pos = trajs[frame]
        sc_part._offsets3d = (pos[:, 0], pos[:, 1], all_z[frame])

        gp = gbests[frame]
        sc_gbest._offsets3d = ([gp[0]], [gp[1]], [all_gz[frame]])

        ax3.view_init(elev=28, azim=float(azims[frame]))

        xs = list(range(frame + 1))
        ys = fit_frames[:frame + 1]
        line_live.set_data(xs, ys)
        dot_live.set_data([frame], [fit_frames[frame]])

        txt3.set_text(f"Frame {frame+1}/{n_frames}\nBest: {fit_frames[frame]:.3e}")
        return sc_part, sc_gbest, line_live, dot_live, txt3

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    _save_or_show(anim, save_path, fps, dpi)
    plt.close(fig)
    return anim