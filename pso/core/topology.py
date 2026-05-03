# @author: Íñigo Martínez Jiménez
# This module defines the topology abstraction used in the PSO.
# The topology determines which "social best" each particle follows during
# the velocity update. In global topology all particles use the same global
# best; other topologies (ring, local) would restrict each particle to a
# subset of neighbors.

from abc import ABC, abstractmethod
import numpy as np


class TopologyStrategy(ABC):
    """
    Abstract base class for swarm topology strategies.

    A topology strategy answers the question: given the current swarm state,
    what is the best known position that each particle should be attracted to?
    The result is called the social best, and has shape (n_particles, dim).
    """

    @abstractmethod
    def get_social_best(self, swarm) -> np.ndarray:
        """
        Return the social best position for each particle.

        Args:
            swarm: Current Swarm object.

        Returns:
            np.ndarray: Social best positions, shape (n_particles, dim).
        """
        ...


class GlobalTopology(TopologyStrategy):
    """
    Global-best topology: every particle is attracted to the single best
    position found by the whole swarm. This is the classic PSO topology
    and converges fast but can get stuck in local optima on multimodal functions.
    """
    def get_social_best(self, swarm) -> np.ndarray:
        # Broadcast the single global best to all particles.
        # np.broadcast_to returns a read-only view with no extra memory.
        return np.broadcast_to(swarm.b_gposition, (swarm.n_particles, swarm.dim))


def choose_topology(name: str) -> TopologyStrategy:
    """
    Return the topology strategy associated with the given name.

    Args:
        name (str): Topology name ('global' is the only one currently implemented).

    Raises:
        ValueError: If the topology name is not recognized.

    Returns:
        TopologyStrategy: Instantiated topology strategy.
    """
    match name:
        case "global":
            return GlobalTopology()
        case _:
            raise ValueError(f"Unknown topology: '{name}'. Available: 'global'.")