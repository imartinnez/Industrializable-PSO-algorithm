from pso.parallel.V0_sequential import VO_sequential
from pso.parallel.V1_threading import V1_threading
from pso.parallel.V2_multiprocessing import V2_multiprocessing



"""lo ideal es que cuando se tenga que elegir como se va a evaluar el pso, no se tenga que tocar el motor
por si se decide añadir mas evaluadores, por lo que se crea esta funcion"""
def choose_evaluator(mode, fitness_f):
    match mode:
        case "sequential":
            return VO_sequential(fitness_f)
        case "threading":
            return V1_threading(fitness_f)
        case "multiprocessing":
            return V2_multiprocessing(fitness_f) #se debe poner tambien el chunksize como parametro
        case "async":
            return
        case "vectorized":
            return
        case _:
            raise ValueError("Invalid mode")