# @author: Íñigo Martínez Jiménez
# This module defines a simple script to generate PSO visualizations
# for several benchmark functions in 2D and 3D, running the optimizer
# and saving the resulting animations to an output directory

from pso.experiments.benchmarks import Instance
from pso.objectives.registry import get_objective
from pso.io.paths import make_run_dir
from pso.viz.animator import animate_2d, animate_3d


if __name__ == "__main__":

    # Objective functions that will be visualized
    OBJECTIVES = ["sphere", "rosenbrock", "rastrigin", "ackley"]

    # Only 2D and 3D are used because those are the dimensions that can be visualized
    DIMS = [2, 3]

    # Frames per second used in the generated animations
    FPS = 15

    # Create one output folder for this visualization run
    outdir = make_run_dir("viz")

    for obj_name in OBJECTIVES:
        obj = get_objective(obj_name)

        for dim in DIMS:
            # Run one PSO instance for the selected objective and dimension
            result = Instance(
                name=f"viz_{obj_name}_d{dim}",
                fitness_f=obj.function,
                dim=dim,
                constraints=obj.constraints,
                seed=42,
                max_iter=300,
                patience=50,
                imp_min=1e-6,
                n_particles=40,
                strategy="clamp",
                fitness_policy="plain",
                topology="global",
                tol=0.0,
                mode="sequential",
            ).run_instance()

            # Output file where the animation will be saved
            save_path = outdir / f"{obj_name}_{dim}d.gif"

            # Use the appropriate animation function depending on the dimension
            if dim == 2:
                animate_2d(
                    result,
                    obj.function,
                    obj.constraints,
                    title=f"PSO – {obj_name} (d=2)",
                    save_path=save_path,
                    fps=FPS
                )
            else:
                animate_3d(
                    result,
                    obj.function,
                    obj.constraints,
                    title=f"PSO – {obj_name} (d=3)",
                    save_path=save_path,
                    fps=FPS
                )