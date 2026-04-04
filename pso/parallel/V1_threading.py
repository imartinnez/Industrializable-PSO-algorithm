import numpy as np
from concurrent.futures import ThreadPoolExecutor

class V1_threading:
    def __init__(self, fitness_f, max_workers=None): # con max_workers=None Python elige automáticamente:  min(32, número_de_cpu + 4)
        self.fitness_f = fitness_f
        self.max_workers = max_workers

    def evaluate(self, positions):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            values = list(executor.map(self.fitness_f, positions))

        return np.array(values, dtype=float)