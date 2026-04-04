import numpy as np
from concurrent.futures import ProcessPoolExecutor

class V2_multiprocessing:
    def __init__(self, fitness_f, max_workers=None, chunksize=5): # con max_workers=None Python elige automáticamente:  min(32, número_de_cpu + 4)
        self.fitness_f = fitness_f
        self.max_workers = max_workers
        self.chunksize = chunksize

    def evaluate(self, positions):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            values = list(executor.map(self.fitness_f, positions, chunksize=self.chunksize))

        return np.array(values, dtype=float)