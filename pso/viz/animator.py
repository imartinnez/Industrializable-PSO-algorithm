import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


# Helpers internos

def _build_grid(fitness_f, constraints, resolution=250):
    """Malla 2D para el fondo de contorno/superficie."""
    low, high = constraints
    xs = np.linspace(low, high, resolution)
    ys = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(xs, ys)
    # vectorize evalúa la función escalar en cada celda de la malla
    Z = np.vectorize(lambda xi, yi: fitness_f(np.array([xi, yi])))(X, Y)
    return X, Y, Z


def _fitness_frames(result):
    """
    Alinea best_fitness_by_iter (una entrada por iteración completa)
    con los frames grabados (que pueden estar submuestreados).
    """
    n_frames = len(result.trajectories)
    n_full   = result.iterations
    step     = max(1, n_full // n_frames)
    return [
        result.best_fitness_by_iter[min(i * step, n_full - 1)]
        for i in range(n_frames)
    ]


def _save_or_show(anim, save_path, fps, dpi):
    if save_path is None:
        plt.show()
        return
    if save_path.endswith(".gif"):
        anim.save(save_path, writer="pillow", fps=fps)
    else:
        anim.save(save_path, writer="ffmpeg", fps=fps, dpi=dpi)
    print(f"Guardado en: {save_path}")


# Animación 2D

def animate_2d(result, fitness_f, constraints,
               title="PSO 2D", save_path=None, fps=15, dpi=120):
    """
    Anima el enjambre sobre un contorno 2D más curva de convergencia.

    result    : Result con record_trajectories=True
    fitness_f : función objetivo callable
    constraints : (low, high)
    save_path : ruta .gif o .mp4  |  None → ventana interactiva
    """
    if result.trajectories is None:
        raise ValueError("Ejecuta PSO con record_trajectories=True primero.")

    trajs      = result.trajectories          # lista de (N, dim)
    gbests     = result.best_positions_by_iter # lista de (dim,)
    fit_frames = _fitness_frames(result)
    n_frames   = len(trajs)
    low, high  = constraints

    # Malla de fondo
    print("Construyendo malla 2D…")
    X, Y, Z = _build_grid(fitness_f, constraints)

    # Figura: dos paneles
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs  = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    ax_s = fig.add_subplot(gs[0])   # panel enjambre
    ax_c = fig.add_subplot(gs[1])   # panel convergencia

    # Panel enjambre
    # Contorno con escala logarítmica si los valores lo permiten
    zmin = max(Z.min(), 1e-10)
    levels = np.exp(np.linspace(np.log(zmin), np.log(Z.max()), 40))
    ax_s.contourf(X, Y, Z, levels=levels, cmap="inferno", alpha=0.85)
    ax_s.contour (X, Y, Z, levels=levels, colors="white",
                  linewidths=0.2, alpha=0.3)

    ax_s.set_xlim(low, high)
    ax_s.set_ylim(low, high)
    ax_s.set_xlabel("x₁")
    ax_s.set_ylabel("x₂")

    # Objetos que se actualizarán en cada frame (empiezan vacíos)
    scat_particles = ax_s.scatter([], [], s=20, c="cyan",
                                  edgecolors="white", linewidths=0.3,
                                  zorder=5, label="Partículas")
    scat_gbest     = ax_s.scatter([], [], s=180, c="red", marker="*",
                                  edgecolors="white", linewidths=0.4,
                                  zorder=6, label="Mejor global")
    ax_s.legend(loc="upper right", fontsize=8)

    # Texto de iteración en la esquina
    txt = ax_s.text(0.02, 0.97, "", transform=ax_s.transAxes,
                    fontsize=9, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    # Panel convergencia
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Mejor fitness")
    ax_c.set_title("Convergencia")
    ax_c.grid(True, alpha=0.3)

    # Curva completa en gris de fondo (referencia)
    ax_c.plot(range(n_frames), fit_frames, color="lightgray", linewidth=0.8)
    # Curva viva que crece con la animación
    line_live, = ax_c.plot([], [], color="royalblue", linewidth=1.8)
    dot_live,  = ax_c.plot([], [], "ro", markersize=6)

    ax_c.set_xlim(0, n_frames)
    pos_vals = [v for v in fit_frames if v > 0]
    if pos_vals:
        ax_c.set_ylim(min(pos_vals) * 0.1, max(fit_frames) * 10)
        ax_c.set_yscale("log")

    # FuncAnimation
    def init():
        scat_particles.set_offsets(np.empty((0, 2)))
        scat_gbest.set_offsets(np.empty((0, 2)))
        line_live.set_data([], [])
        dot_live.set_data([], [])
        txt.set_text("")
        return scat_particles, scat_gbest, line_live, dot_live, txt

    def update(frame):
        # Posiciones de las partículas (solo x1, x2)
        pos = trajs[frame][:, :2]
        scat_particles.set_offsets(pos)

        # Mejor global
        gp = gbests[frame][:2]
        scat_gbest.set_offsets([[gp[0], gp[1]]])

        # Curva de convergencia hasta el frame actual
        xs = list(range(frame + 1))
        ys = fit_frames[:frame + 1]
        line_live.set_data(xs, ys)
        dot_live.set_data([frame], [fit_frames[frame]])

        txt.set_text(f"Frame {frame+1}/{n_frames}\nBest: {fit_frames[frame]:.3e}")
        return scat_particles, scat_gbest, line_live, dot_live, txt

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=1000 // fps,
        blit=True,          # blit=True funciona bien en 2D → más fluido
    )

    _save_or_show(anim, save_path, fps, dpi)
    plt.close(fig)
    return anim


# Animación 3D

def animate_3d(result, fitness_f, constraints,
               title="PSO 3D", save_path=None, fps=12, dpi=110):
    """
    Anima el enjambre sobre una superficie 3D.
    Las partículas se plotean en (x1, x2, f(x1,x2)) → flotan sobre la malla.
    La cámara rota lentamente para mostrar profundidad.
    """
    if result.trajectories is None:
        raise ValueError("Ejecuta PSO con record_trajectories=True primero.")

    trajs      = result.trajectories
    gbests     = result.best_positions_by_iter
    fit_frames = _fitness_frames(result)
    n_frames   = len(trajs)
    low, high  = constraints

    print("Construyendo malla 3D…")
    X, Y, Z = _build_grid(fitness_f, constraints, resolution=60)

    # Figura
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs  = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax_c = fig.add_subplot(gs[1])

    # Superficie (estática)
    ax3.plot_surface(X, Y, Z, cmap="inferno", alpha=0.5,
                     linewidth=0, antialiased=True)
    ax3.set_xlabel("x₁", fontsize=8)
    ax3.set_ylabel("x₂", fontsize=8)
    ax3.set_zlabel("f(x)", fontsize=8)
    ax3.view_init(elev=28, azim=-60)    # ángulo inicial

    # Precalcula los valores z de todas las partículas en todos los frames
    # para no recalcular dentro de update() (lento)
    print("Precalculando fitness de partículas…")
    all_z     = [np.array([fitness_f(p) for p in trajs[i]]) for i in range(n_frames)]
    all_gz    = [fitness_f(gbests[i])                        for i in range(n_frames)]

    # Scatter inicial (se sobreescribirá en frame 0)
    pos0 = trajs[0]
    sc_part  = ax3.scatter(pos0[:,0], pos0[:,1], all_z[0],
                           s=15, c="cyan", alpha=0.7, depthshade=True)
    gp0 = gbests[0]
    sc_gbest = ax3.scatter([gp0[0]], [gp0[1]], [all_gz[0]],
                           s=140, c="red", marker="*", zorder=6)

    txt3 = ax3.text2D(0.02, 0.95, "", transform=ax3.transAxes,
                      fontsize=8, va="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    # Panel convergencia
    ax_c.set_xlabel("Frame")
    ax_c.set_ylabel("Mejor fitness")
    ax_c.set_title("Convergencia")
    ax_c.grid(True, alpha=0.3)
    ax_c.plot(range(n_frames), fit_frames, color="lightgray", linewidth=0.8)
    line_live, = ax_c.plot([], [], color="royalblue", linewidth=1.8)
    dot_live,  = ax_c.plot([], [], "ro", markersize=6)
    ax_c.set_xlim(0, n_frames)
    pos_vals = [v for v in fit_frames if v > 0]
    if pos_vals:
        ax_c.set_ylim(min(pos_vals) * 0.1, max(fit_frames) * 10)
        ax_c.set_yscale("log")

    # FuncAnimation
    # blit=False obligatorio en 3D (Axes3D no lo soporta)
    azims = np.linspace(-60, 300, n_frames)   # rotación completa de la cámara

    def update(frame):
        pos = trajs[frame]
        sc_part._offsets3d  = (pos[:,0], pos[:,1], all_z[frame])

        gp = gbests[frame]
        sc_gbest._offsets3d = ([gp[0]], [gp[1]], [all_gz[frame]])

        ax3.view_init(elev=28, azim=float(azims[frame]))

        xs = list(range(frame + 1))
        ys = fit_frames[:frame + 1]
        line_live.set_data(xs, ys)
        dot_live.set_data([frame], [fit_frames[frame]])

        txt3.set_text(f"Frame {frame+1}/{n_frames}\nBest: {fit_frames[frame]:.3e}")
        return sc_part, sc_gbest, line_live, dot_live, txt3

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    _save_or_show(anim, save_path, fps, dpi)
    plt.close(fig)
    return anim