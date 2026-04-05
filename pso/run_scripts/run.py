from pyswarm import pso
import numpy as np
import os


import pso.objectives.registry as o
import pso.experiments.benchmarks as i
import pso.viz.animator as a

# faltan las estrategias explicitas (clamp/reflect/penalty, elegida y documentada)
# faltan criterios de parada (iteraciones, tolerancia, estancamiento)
# arreglar visualizaciones y entden, poner para almacenar experimentos (timer en el nombre)
# grid search
# analisis

# ── Configuración de visualización ───────────────────────────────────────────
VISUALIZE        = True    # ← Cambia esto a False para saltar la animación
VIZ_OBJECTIVE    = "rastrigin"
VIZ_FPS          = 15

if __name__ == "__main__":

    sphere = o.get_objective("sphere")

    instance1 = i.Instance(
        name="sphere_d10",
        fitness_f=sphere.function,
        dim=10,
        constraints=sphere.constraints,
        seed=1,
        max_iter=2000,
        n_particles=50,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        tol=0.0,
        mode="sequential"
    )

    instance2 = i.Instance(
        name="sphere2_d10",
        fitness_f=sphere.function,
        dim=10,
        constraints=sphere.constraints,
        seed=1,
        max_iter=2000,
        n_particles=50,
        strategy="clamp",
        fitness_policy="plain",
        topology="global",
        tol=0.0,
        mode="threading"
    )
    
    result1 = instance1.run_instance()

    print("PSO result:")
    print("best value:", result1.b_value)
    print("total time:", result1.total_time)
    print("fitness eval total:", result1.fitness_eval_time_total)
    print("iterations:", result1.iterations)
    #print("curve:", result.best_fitness_by_iter)

    result2 = instance1.run_instance()

    print("PSO 2 result:")
    print("best value 2:", result2.b_value)
    print("total time 2:", result2.total_time)
    print("fitness eval total 2:", result2.fitness_eval_time_total)
    print("iterations 2:", result2.iterations)
    #print("curve:", result.best_fitness_by_iter)


    # pyswarm
    lb = [sphere.constraints[0]] * 10
    ub = [sphere.constraints[1]] * 10

    np.random.seed(1)
    xopt, fopt = pso(
        sphere.function,
        lb,
        ub,
        swarmsize=50,
        omega=0.7,
        phip=1.5,
        phig=1.5,
        maxiter=2000,
        minfunc=1e-50,
        minstep=1e-50
)

    print("\nPySwarm result: ")
    print(fopt)






    # ── Visualización (al final) ─────────────────────────────────────────────────
    if VISUALIZE:

        os.makedirs("results", exist_ok=True)
        obj = o.get_objective(VIZ_OBJECTIVE)

        # 2D
        print("\n=== Visualización 2D ===")
        result_2d = i.Instance(
            name="viz_2d", fitness_f=obj.function, dim=2,
            constraints=obj.constraints, seed=1, max_iter=300,
            n_particles=40, strategy="inertia", topology="global",
            tol=0.0, mode="sequential",
        ).run_instance()

        a.animate_2d(
            result=result_2d, fitness_f=obj.function, constraints=obj.constraints,
            title=f"PSO – {VIZ_OBJECTIVE} (d=2)",
            save_path=f"results/viz_{VIZ_OBJECTIVE}_2d.gif", fps=VIZ_FPS,
        )

        # 3D
        print("\n=== Visualización 3D ===")
        result_3d = i.Instance(
            name="viz_3d", fitness_f=obj.function, dim=3,
            constraints=obj.constraints, seed=1, max_iter=300,
            n_particles=40, strategy="inertia", topology="global",
            tol=0.0, mode="sequential",
        ).run_instance()

        a.animate_3d(
            result=result_3d, fitness_f=obj.function, constraints=obj.constraints,
            title=f"PSO – {VIZ_OBJECTIVE} (d=3)",
            save_path=f"results/viz_{VIZ_OBJECTIVE}_3d.gif", fps=VIZ_FPS,
        )