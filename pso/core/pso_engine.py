import numpy as np
from time import perf_counter

import pso.core.swarm as s
import pso.core.result as r

class PSO:
    def __init__(self, n_particles, fitness_f, dim, constraints, strategy, fitness_policy, topology, tol, max_iter, patience, imp_min, w, c1, c2, optimum_value, evaluator):
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
        self.penalty_lambda=float(1e4)

        if self.fitness_policy == "penalty" and self.strategy in ["clamp", "reflect"]:
            print("Strategy changed to 'None' because of the use of Penalty Policy")
            self.strategy = "None"
    
    def generate_particles(self):
        low, high = self.constraints
        vmax = 0.2 * (high - low)
        
        positions = np.random.uniform(low, high, size=(self.n_particles, self.dim))
        velocities = np.random.uniform(-vmax, vmax, size=(self.n_particles, self.dim))
        
        self.swarm = s.Swarm(positions, velocities, self.dim, self.constraints, self.strategy)
        
        values = self.evaluator.evaluate(self.swarm.positions)
        values = self.apply_fitness_policy(values)
        self.swarm.initialize_bests_from_values(values)

    def penalty_vector(self):
        """
        Penalización por violar bounds (vector N):
        sum_j [ max(0, x-ub)^2 + max(0, lb-x)^2 ]
        """
        excess_high = np.maximum(0.0, self.swarm.positions - self.swarm.upper_bounds)
        excess_low  = np.maximum(0.0, self.swarm.lower_bounds - self.swarm.positions)

        p = (excess_high**2 + excess_low**2).sum(axis=1)
        return p

    def apply_fitness_policy(self, values):
        match self.fitness_policy:
            case "plain":
                return values
            case "penalty":
                return values + self.penalty_lambda * self.penalty_vector()
            case _:
                raise ValueError("Invalid fitness policy")
    
    def run(self, seed=42):
        pso_start = perf_counter()
        np.random.seed(seed)

        self.generate_particles()
        seeds = np.random.randint(1, 2**30, self.max_iter)
        b_global = float("inf")

        best_fitness_by_iter = []
        fitness_eval_time_by_iter = []
        iterations = 0
        counter = 0
        trajectories = [] if self.dim <= 3 else None # No tiene sentido almacenar trayectorias para dimensiones mayores que 3, ya que no se puede visualizar
        best_positions_by_iter = []


        for i in range(self.max_iter):
            
            seed = seeds[i]
            np.random.seed(seed)
            
            #1- Se evalua el fitness midiendo el tiempo que tarda
            eval_start = perf_counter()
            values = self.evaluator.evaluate(self.swarm.positions)
            values = self.apply_fitness_policy(values)
            eval_time = perf_counter() - eval_start
            fitness_eval_time_by_iter.append(eval_time)

            self.swarm.update_personal_bests(values)
            
            #2- Se actualiza el mejor global
            self.swarm.update_b_global()
            best_fitness_by_iter.append(float(self.swarm.b_gvalue))

            #2.5- Se registran los datos
            if trajectories is not None:
                trajectories.append(self.swarm.positions.copy())

            best_positions_by_iter.append(self.swarm.b_gposition.copy()) # ??????
            
            #3- Se mueven las particulas
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            #mejor punto global repetido para todas las partículas, para poder hacer la resta vectorizada con sus posiciones actuales.
            b_global_matrix = np.broadcast_to(self.swarm.b_gposition, (self.n_particles, self.dim))
            
            self.swarm.update_velocities(self.w, self.c1, self.c2, r1, r2, b_global_matrix)
            self.swarm.update_positions()

            iterations = i + 1
            
            #4- Criterios de parada
            if abs(b_global - self.swarm.b_gvalue) < self.imp_min:
                counter += 1
            else:
                counter = 0
            
            if counter >= self.patience:
                break

            if self.optimum_value is not None:
                if abs(self.swarm.b_gvalue - self.optimum_value) < self.tol:
                    break

            b_global = self.swarm.b_gvalue
            
        total_time = perf_counter() - pso_start
        
        result = r.Result(
            self.swarm.b_gposition.copy(),
            float(self.swarm.b_gvalue),
            total_time,
            sum(fitness_eval_time_by_iter),
            fitness_eval_time_by_iter,
            best_fitness_by_iter,
            iterations,
            best_positions_by_iter
        )

        return result