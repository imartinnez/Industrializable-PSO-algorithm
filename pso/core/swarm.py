class Swarm:
    def __init__(self, particles):
        self.particles = particles
        self.b_gposition = None
        self.b_gvalue = float("inf")
    
    def update_b_global(self):
        # Para evitar errores y no guardar posiciones antiguas
        self.b_gposition = None
        self.b_gvalue = float("inf")
        for particle in self.particles:
            if(particle.best_value < self.b_gvalue):
                self.b_gvalue = particle.best_value
                self.b_gposition = particle.position.copy()