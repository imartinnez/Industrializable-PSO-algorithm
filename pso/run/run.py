from pyswarm import pso

import pso.objectives.registry as o
import pso.experiments.benchmarks as i

if __name__ == "__main__":

    sphere = o.get_objective("sphere")

    instance1 = i.Instance(
        name="sphere_d10",
        objective=sphere.function,
        dim=10,
        constraints=sphere.constraints,
        seed=1,
        max_iter=2000,
        n_particles=50,
        strategy="inertia",
        topology="global",
        tol=0.0
    )
    
    best_pos, best_val = instance1.run_instance()

    print("PSO result: ")
    print(best_val)


    # pyswarm
    lb = [sphere.constraints[0]] * 10
    ub = [sphere.constraints[1]] * 10

    xopt, fopt = pso(
        sphere.function,
        lb,
        ub,
        swarmsize=50,
        maxiter=2000,
        minfunc=1e-50,
        minstep=1e-50
    )

    print("\nPySwarm result: ")
    print(fopt)