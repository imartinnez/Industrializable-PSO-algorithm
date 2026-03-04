import numpy as np

import core.particle as prt
import core.swarm as s

class PSO:
    def __init__(self, n_particles, fitness_f, d, constraints, strategy, w, c1, c2,max_iter, topology = "global", tol = 0.0):
        self.n_particles = n_particles
        self.fitness_f = fitness_f
        self.d = d
        self.constraints = constraints
        self.strategy = strategy
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.topology = topology
        self.tol = tol
        self.swarm = None
    
    def generate_particles(self):
        particles = []
        for _  in range(self.n_particles):
            position = np.array([np.random.uniform(min_i, max_i) for (min_i, max_i) in self.constraints])
            velocity = np.array([np.random.uniform(-abs(max_i-min_i), abs(max_i-min_i)) for (min_i,max_i) in self.constraints])
            value = self.fitness_f(position)
        
            p = prt.Particle(position, velocity, value)
            particles.append(p)
    
        self.swarm = s.Swarm(particles)
        self.swarm.update_b_global()
    
    def run(self):
        self.generate_particles()
        np.random.seed(42)
        seeds = np.random.randint(1, 2**30, self.max_iter)
        b_global = float("inf")
        
        for i in range(self.max_iter):
            
            seed = seeds[i]
            np.random.seed(seed)
            
            for p in self.swarm.particles:   
                
                value = self.fitness_f(p.position)
                p.update_best_value(value)
            
            self.swarm.update_b_global()
            
            for p in self.swarm.particles:   
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                p.update_velocity(self.w, self.c1, self.c2, r1, r2, self.swarm.b_gposition)
                p.update_position()
                
            if abs(b_global - self.swarm.b_gvalue) < self.tol:
                break
            b_global = self.swarm.b_gvalue
            

        
        
    
class PSOSequential(PSO):
    def evaluate(self):
        pass

class PSOParallel(PSO):
    def evaluate(self):
        pass