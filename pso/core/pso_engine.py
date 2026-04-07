# @author: Íñigo Martínez Jiménez
# This module defines the PSO class used to run the optimization process


import numpy as np
from time import perf_counter
from collections.abc import Callable
from typing import Any

import pso.core.swarm as s
import pso.core.result as r

class PSO:
    """
    This class manages the full PSO execution, including particle initialization,
    fitness evaluation, best value updates, stopping criteria, and final result storage.
    """
    def __init__(self, n_particles: int, fitness_f: Callable[[np.ndarray], float], 
                 dim: int, constraints: tuple[float, float], strategy: str, fitness_policy: str, 
                 topology: str, tol: float, max_iter: int, patience: int, imp_min: float, 
                 w: float, c1: float, c2: float, optimum_value: float | None, evaluator: Any) -> None:
        """
        Initialize the PSO object with the main optimization settings.

        Args:
            n_particles (int): Number of particles in the swarm.
            fitness_f (Callable[[np.ndarray], float]): Objective function to minimize.
            dim (int): Dimension of the function.
            constraints (tuple[float, float]): Lower and upper bounds of the search space.
            strategy (str): Boundary handling strategy.
            fitness_policy (str): Fitness evaluation policy.
            topology (str): Swarm topology.
            tol (float): Tolerance used for early stopping near the optimum.
            max_iter (int): Maximum number of iterations.
            patience (int): Number of iterations allowed without meaningful improvement.
            imp_min (float): Minimum improvement required to reset the patience counter.
            w (float): Inertia coefficient.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            optimum_value (float | None): Known optimum value of the function, if available.
            evaluator (Any): Evaluator object used to compute particle fitness values.
        """
        self.n_particles = n_particles
        self.fitness_f = fitness_f
        self.dim = dim
        self.constraints = constraints
        self.strategy = strategy
        self.fitness_policy = fitness_policy
        self.topology = topology
        self.tol = tol
        self.max_iter = max_iter
        self.patience = patience
        self.imp_min = imp_min
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.optimum_value = optimum_value
        self.evaluator = evaluator
        self.swarm = None
        self.penalty_lambda = float(1e4)

        # Penalty policy should not be mixed with clamp or reflect strategies
        # because the penalty already handles out-of-bounds solutions
        if self.fitness_policy == "penalty" and self.strategy in ["clamp", "reflect"]:
            print("Strategy changed to 'None' because of the use of Penalty Policy")
            self.strategy = "None"
    
    def generate_particles(self) -> None:
        """
        Generate the initial swarm and evaluate its first fitness values.
        """
        low, high = self.constraints
        vmax = 0.2 * (high - low)  # max velocity based on search range

        # initialize random positions and velocities
        positions = np.random.uniform(low, high, size=(self.n_particles, self.dim))
        velocities = np.random.uniform(-vmax, vmax, size=(self.n_particles, self.dim))

        # create swarm with initial state
        self.swarm = s.Swarm(positions, velocities, self.dim, self.constraints, self.strategy)

        # evaluate fitness and set initial bests
        values = self.evaluator.evaluate(self.swarm.positions)
        values = self.apply_fitness_policy(values)
        self.swarm.initialize_bests_from_values(values)

    def penalty_vector(self) -> np.ndarray:
        """
        Compute the penalty applied to particles that violate the bounds.

        Returns:
            np.ndarray: Penalty value for each particle.
        """
        # compute how much each particle exceeds upper and lower bounds
        excess_high = np.maximum(0.0, self.swarm.positions - self.swarm.upper_bounds)
        excess_low = np.maximum(0.0, self.swarm.lower_bounds - self.swarm.positions)

        # penalty is the sum of squared violations per particle
        penalty = (excess_high**2 + excess_low**2).sum(axis=1)
        return penalty

    def apply_fitness_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Apply the selected fitness policy to the evaluated values.

        Args:
            values (np.ndarray): Raw fitness values.

        Raises:
            ValueError: If the fitness policy is not valid.

        Returns:
            np.ndarray: Fitness values after applying the selected policy.
        """
        match self.fitness_policy:
            case "plain":
                return values
            case "penalty":
                return values + self.penalty_lambda * self.penalty_vector()
            case _:
                raise ValueError("Invalid fitness policy")
    
    def run(self, seed: int = 42) -> r.Result:
        """
        Run the PSO algorithm and return the final result.

        Args:
            seed (int): Random seed used to make the execution reproducible.

        Returns:
            r.Result: Final result of the PSO execution.
        """
        pso_start = perf_counter()
        np.random.seed(seed)

        self.generate_particles()  # initialize swarm with random positions/velocities
        seeds = np.random.randint(1, 2**30, self.max_iter)  # different seed per iteration
        b_global = float("inf")

        best_fitness_by_iter = []
        fitness_eval_time_by_iter = []
        iterations = 0
        counter = 0  # used for early stopping (no improvement)

        # store trajectories only if dimension is small, does not make sense more than 3 dimensions for plotting
        trajectories = [] if self.dim <= 3 else None
        best_positions_by_iter = []

        for i in range(self.max_iter):
            np.random.seed(seeds[i])  # make each iteration reproducible

            # evaluate fitness and measure evaluation time
            eval_start = perf_counter()
            values = self.evaluator.evaluate(self.swarm.positions)
            values = self.apply_fitness_policy(values)  # apply penalties if needed
            fitness_eval_time_by_iter.append(perf_counter() - eval_start)

            self.swarm.update_personal_bests(values)  # update each particle's best

            # update global best among all particles
            self.swarm.update_b_global()
            best_fitness_by_iter.append(float(self.swarm.b_gvalue))

            # save positions/history for analysis
            if trajectories is not None:
                trajectories.append(self.swarm.positions.copy())
            best_positions_by_iter.append(self.swarm.b_gposition.copy())

            # generate random factors and update movement
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            b_global_matrix = np.broadcast_to(self.swarm.b_gposition, (self.n_particles, self.dim))

            self.swarm.update_velocities(self.w, self.c1, self.c2, r1, r2, b_global_matrix)
            self.swarm.update_positions()

            iterations = i + 1

            # early stopping if improvement is too small for several iterations
            if abs(b_global - self.swarm.b_gvalue) < self.imp_min:
                counter += 1
            else:
                counter = 0
            if counter >= self.patience:
                break

            # stop if known optimum is reached within tolerance
            if self.optimum_value is not None and abs(self.swarm.b_gvalue - self.optimum_value) < self.tol:
                break

            b_global = self.swarm.b_gvalue  # update reference best

        total_time = perf_counter() - pso_start
        
        # return final result with best solution and collected metrics
        result = r.Result(
            self.swarm.b_gposition.copy(),
            float(self.swarm.b_gvalue),
            total_time,
            sum(fitness_eval_time_by_iter),
            fitness_eval_time_by_iter,
            best_fitness_by_iter,
            iterations,
            best_positions_by_iter,
            trajectories
        )

        return result