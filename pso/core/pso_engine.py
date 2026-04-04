import numpy as np
from time import perf_counter

import pso.core.swarm as s
import pso.core.result as r

class PSO:
    def __init__(self, n_particles, fitness_f, dim, constraints, strategy, topology, tol, max_iter, w, c1, c2, evaluator):
        self.n_particles = n_particles
        self.fitness_f = fitness_f
        self.dim = dim
        self.constraints = constraints
        self.strategy = strategy
        self.topology = topology
        self.tol = tol
        self.max_iter = max_iter
        self.evaluator = evaluator
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = None
    
    def generate_particles(self):
        low, high = self.constraints
        vmax = 0.2 * (high - low)
        
        positions = np.random.uniform(low, high, size=(self.n_particles, self.dim))
        velocities = np.random.uniform(-vmax, vmax, size=(self.n_particles, self.dim))
        
        self.swarm = s.Swarm(positions, velocities, self.dim, self.constraints)
        
        values = self.evaluator.evaluate(self.swarm.positions)
        self.swarm.initialize_bests_from_values(values)
    
    def run(self, seed=42):
        pso_start = perf_counter()
        np.random.seed(seed)

        self.generate_particles()
        seeds = np.random.randint(1, 2**30, self.max_iter)
        b_global = float("inf")

        best_fitness_by_iter = []
        fitness_eval_time_by_iter = []
        iterations = 0
        trajectories = []
        best_positions_by_iter = []


        for i in range(self.max_iter):
            
            seed = seeds[i]
            np.random.seed(seed)
            
            #1- Se evalua el fitness midiendo el tiempo que tarda
            eval_start = perf_counter()
            values = self.evaluator.evaluate(self.swarm.positions)
            eval_time = perf_counter() - eval_start
            fitness_eval_time_by_iter.append(eval_time)

            self.swarm.update_personal_bests(values)
            
            #2- Se actualiza el mejor global
            self.swarm.update_b_global()
            best_fitness_by_iter.append(float(self.swarm.b_gvalue))

            #2.5- Se registran los datos
            trajectories.append(self.swarm.positions.copy())  # ??????
            best_positions_by_iter.append(self.swarm.b_gposition.copy()) # ??????
            
            #3- Se mueven las particulas
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            #mejor punto global repetido para todas las partículas, para poder hacer la resta vectorizada con sus posiciones actuales.
            b_global_matrix = np.broadcast_to(self.swarm.b_gposition, (self.n_particles, self.dim))
            
            self.swarm.update_velocities(self.w, self.c1, self.c2, r1, r2, b_global_matrix)
            self.swarm.update_positions()

            iterations = i + 1
            
            #4- Criterio de parada
            if abs(b_global - self.swarm.b_gvalue) < self.tol:
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
            trajectories,
            best_positions_by_iter
        )

        return result