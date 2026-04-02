import numpy as np

class VO_sequential:
    def __init__(self, fitness_f):
        self.fitness_f = fitness_f

    def evaluate(self, positions):
        return np.array([self.fitness_f(position) for position in positions])