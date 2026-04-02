from pso.parallel.V0_sequential import VO_sequential

"""lo ideal es que cuando se tenga que elegir como se va a evaluar el pso, no se tenga que tocar el motor
por si se decide añadir mas evaluadores, por lo que se crea esta funcion"""
def choose_evaluator(mode, fitness_f):
    match mode:
        case "sequential":
            return VO_sequential(fitness_f)
        case "threading":
            return
        case "multiprocessing":
            return
        case "async":
            return
        case "vectorized":
            return
        case _:
            raise ValueError("Invalid mode")