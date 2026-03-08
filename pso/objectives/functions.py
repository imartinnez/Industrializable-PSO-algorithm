import numpy as np

def sphere(x: np.array) -> float:
    return float(np.sum(x ** 2))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x: np.ndarray) -> float:
    d = len(x)
    return float(10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
                 + 20 + np.e)
    
    
#resto de cosas experimentales, sobre todo semillas, dimensiones e instancias reproducibles, lo llevaría a experiments/