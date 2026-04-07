# @author: Íñigo Martínez Jiménez
# This module defines the Swarm class used in the PSO

"""
IMPORTANT: Read before proceeding

Although the PSO could have been implemented as a list of Particle objects,
in this project it was decided to represent the swarm using NumPy matrices and
vectors within a Swarm class, as this approach better aligned
with the architecture required by the problem statement.
The main reason is not simply that NumPy is faster, but that the
problem statement requires maintaining a common core of the algorithm and changing only the
evaluation strategy depending on the version. Therefore, it is cleaner and
more maintainable to work with a homogeneous global state of the swarm (positions,
velocities, personal bests, and global best) rather than distributing that
state among multiple individual objects.
Furthermore, this representation makes it easier for the fitness evaluator to
directly receive a block of particles and apply different
concurrency strategies without having to modify the main engine.
At the same time, storing the swarm in matrices does not imply that the
implementation already corresponds to the vectorized V4 version: it is one thing to store
the state in arrays and quite another for the entire evaluation and update
to be performed in a fully vectorized manner, without Python loops per
particle. In this case, the matrix structure provides a more
coherent design foundation, improves integration with persistence, visualization, and benchmarking,
and paves the way for more complete vectorization in
future versions.
"""

import numpy as np

class Swarm:
    """
    The class stores the full state of the swarm, including particle positions,
    velocities, personal bests, and the global best solution found so far. It also
    handles boundary strategies and the main state updates performed during the run.
    """
    def __init__(self, positions: np.ndarray, velocities: np.ndarray, 
                 dim: int, constraints: tuple[float, float], strategy: str) -> None:
        """
        Initialize the swarm with particle positions, velocities, and search space settings.

        Args:
            positions (np.ndarray): Initial positions of the particles.
            velocities (np.ndarray): Initial velocities of the particles.
            dim (int): Dimension of the function.
            constraints (tuple[float, float]): Lower and upper bounds of the search space.
            strategy (str): Boundary handling strategy to apply after position updates.
        """
        # Particles are stored as NumPy arrays instead of Python objects
        # This avoids unnecessary loops and attribute access
        self.positions = np.asarray(positions, dtype=float)   # (Number of particles, Dimension)
        self.velocities = np.asarray(velocities, dtype=float) # (Number of particles, Dimension)
        self.dim = dim
        self.pbest_positions = positions.copy() # (Number of particles, Dimension)
        self.pbest_values = np.full(positions.shape[0], np.inf, dtype=float)  # (Number of particles,)
        self.b_gposition = np.zeros(self.dim, dtype=float)
        self.b_gvalue = np.inf
        self.n_particles, self.dim = self.positions.shape
        
        low, high = constraints
        self.lower_bounds = np.full(self.dim, low, dtype=float)
        self.upper_bounds = np.full(self.dim, high, dtype=float)

        self.strategy = strategy
        self.current_values = np.full(self.n_particles, np.inf, dtype=float)
    
    def clamp_strategy(self) -> None:
        """
        Keep particle positions inside the search bounds by clipping each coordinate.
        """
        np.clip(self.positions, self.lower_bounds, self.upper_bounds, out=self.positions)

    def reflect_strategy(self) -> None:
        """
        Reflect particles back into the valid range when they cross the bounds.
        This implementation only handles a single rebound.
        """
        mask_high = self.positions > self.upper_bounds
        mask_low = self.positions < self.lower_bounds

        self.positions[mask_high] = (
            self.upper_bounds[mask_high]
            - (self.positions[mask_high] - self.upper_bounds[mask_high])
        )

        self.positions[mask_low] = (
            self.lower_bounds[mask_low]
            + (self.lower_bounds[mask_low] - self.positions[mask_low])
        )

        # Reverse the velocity only in the dimensions that hit the bounds
        self.velocities[mask_high | mask_low] *= -1

        # Extra safety step in case a particle moved too far
        np.clip(self.positions, self.lower_bounds, self.upper_bounds, out=self.positions)

    def update_personal_bests(self, values: np.ndarray) -> None:
        """
        Update each particle's personal best if the current value is better.

        Args:
            values (np.ndarray): Current fitness values of all particles.
        """
        # Copies the actual fitness value of all the particles
        self.current_values = values.copy()
        # Boolean mask that indicates whether each particle has improved its current position or not
        mask_improved = values < self.pbest_values

        # We update the ones that have improved
        self.pbest_values[mask_improved] = values[mask_improved]
        self.pbest_positions[mask_improved] = self.positions[mask_improved]

    def update_b_global(self) -> None:
        """
        Update the global best solution found by the swarm.
        """
        index = np.argmin(self.pbest_values)
        value = self.pbest_values[index]

        if value < self.b_gvalue:
            self.b_gvalue = value
            self.b_gposition = self.pbest_positions[index].copy()

    def update_velocities(self, w: float, c1: float, c2: float, r1: np.ndarray, 
                          r2: np.ndarray, social_best: np.ndarray) -> None:
        """
        Update particle velocities using the standard PSO rule.

        Args:
            w (float): Inertia coefficient.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            r1 (np.ndarray): Random values for the cognitive term.
            r2 (np.ndarray): Random values for the social term.
            social_best (np.ndarray): Best social position used in the update.
        """
        self.velocities = (w * self.velocities) + (c1 * r1 *(self.pbest_positions - self.positions)) + (c2 * r2 *(social_best - self.positions))

    def update_positions(self) -> None:
        """
        Update particle positions and apply the selected boundary handling strategy.

        Raises:
            ValueError: If the selected strategy is not valid.
        """
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

    def initialize_bests_from_values(self, values: np.ndarray) -> None:
        """
        Initialize personal and global best values from the first fitness evaluation.

        Args:
            values (np.ndarray): Initial fitness values of all particles.
        """
        self.update_personal_bests(values)
        self.update_b_global()