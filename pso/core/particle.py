class Particle:
    def __init__(self, position, velocity, value):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_value = value
    
    def update_position(self):
        self.position = self.position + self.velocity
    
    def update_velocity(self, w, c1, c2, r1, r2, b_glb_position):
        self.velocity = (w * self.velocity) + (c1 * r1 *(self.best_position - self.position)) + (c2 * r2 *(b_glb_position - self.position))
    
    def update_best_value(self, value):
        if(value < self.best_value):
            self.best_value = value
            self.best_position = self.position.copy()