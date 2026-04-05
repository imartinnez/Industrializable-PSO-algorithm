import numpy as np

class Swarm:
    def __init__(self, positions, velocities, dim, constraints, strategy):
        # si almacenarmaos las particulas en un array de objetos, tendriamos que hacer muchos bucles, lecturas y escrituras de atributos, mucha busqueda de memoria
        # todo esto se volvería dificil de implementar sin duplicar codigo y queremos un PSO con distintas versiones y modular
        self.positions = np.asarray(positions, dtype=float)   # (Numero particulas, Dimension)
        self.velocities = np.asarray(velocities, dtype=float)  # (N, D)
        self.dim = dim
        self.pbest_positions = positions.copy()     # (N, D)
        self.pbest_values = np.full(positions.shape[0], np.inf, dtype=float)  # (N,)
        self.b_gposition = np.zeros(self.dim, dtype=float)
        self.b_gvalue = np.inf
        self.n_particles, self.dim = self.positions.shape
        
        low, high = constraints
        self.lower_bounds = np.full(self.dim, low, dtype=float)
        self.upper_bounds = np.full(self.dim, high, dtype=float)

        self.strategy = strategy
        
        #para hacer debug
        self.current_values = np.full(self.n_particles, np.inf, dtype=float)
    
    def clamp_strategy(self):
        #Para que ninguna partícula se salga de los límites del problema.
        #recorta cada coordenada de self.positions para que quede entre los limites
        np.clip(self.positions, self.lower_bounds, self.upper_bounds, out=self.positions)

    def reflect_strategy(self):
        # No implementa múltiples rebotes
        # Máscara de los que se pasan por arriba
        mask_high = self.positions > self.upper_bounds
        # Máscara de los que se pasan por abajo
        mask_low = self.positions < self.lower_bounds

        # Reflejar posiciones
        self.positions[mask_high] = (
            self.upper_bounds[mask_high] -
            (self.positions[mask_high] - self.upper_bounds[mask_high])
        )

        self.positions[mask_low] = (
            self.lower_bounds[mask_low] +
            (self.lower_bounds[mask_low] - self.positions[mask_low])
        )

        # Invertir velocidad SOLO en esas dimensiones
        self.velocities[mask_high | mask_low] *= -1

        # (Opcional de seguridad) Asegura que queda dentro por si hubo salto enorme: un solo rebote
        np.clip(self.positions, self.lower_bounds, self.upper_bounds, out=self.positions)
        

    def update_personal_bests(self, values):
        self.current_values = values.copy()  #copia el fitness actual de todas las particulas
        mask_improved = values < self.pbest_values #mascara booleana que dice si cada particula ha mejorado su posicion actual o no
        
        #actualizamos las que han mejorado
        self.pbest_values[mask_improved] = values[mask_improved]
        self.pbest_positions[mask_improved] = self.positions[mask_improved]

    def update_b_global(self):
        index = np.argmin(self.pbest_values) #mejor resultado de todo el swarm
        value = self.pbest_values[index]

        if value < self.b_gvalue:
            self.b_gvalue = value
            self.b_gposition = self.pbest_positions[index].copy()

    def update_velocities(self, w, c1, c2, r1, r2, social_best):
        self.velocities = (w * self.velocities) + (c1 * r1 *(self.pbest_positions - self.positions)) + (c2 * r2 *(social_best - self.positions))

    def update_positions(self):
        self.positions += self.velocities

        match self.strategy:
            case "clamp":
                return self.clamp_strategy()
            case "reflect":
                return self.reflect_strategy()
            case "None":
                return 
            case _:
                raise ValueError("Invalid strategy")

    def initialize_bests_from_values(self, values):
        self.update_personal_bests(values)
        self.update_b_global()