# @author: Íñigo Martínez Jiménez
# This module defines the async evaluator used in the PSO.
# It uses asyncio.gather() to evaluate all particles concurrently, which is
# useful when the fitness function involves variable-latency I/O. All 
# particles "wait" at the same time instead of one
# after another, so total evaluation time approaches the slowest single call
# rather than the sum of all calls.

import asyncio
import random
from collections.abc import Callable

import numpy as np


class V3_async:
    """
    Async evaluator for particle fitness values using asyncio.gather().

    This evaluator is designed for fitness functions that involve I/O latency,
    not for pure CPU-bound functions. The latency is simulated here via
    asyncio.sleep(), but in a real scenario it would be replaced by an actual
    async I/O call.

    With asyncio, all particle evaluations are launched as coroutines and
    suspended at the await point simultaneously. The event loop resumes each
    one as soon as its latency is over. This means 50 particles with 10ms
    latency each take aprox. 10ms total instead of 500ms sequentially.
    """

    def __init__(self, fitness_f: Callable[[np.ndarray], float],
                 latency_range: tuple[float, float] = (0.005, 0.02)) -> None:
        """
        Initialize the evaluator with the objective function and latency settings.

        Args:
            fitness_f (Callable[[np.ndarray], float]): Objective function to evaluate.
            latency_range (tuple[float, float]): Min and max simulated I/O latency
                in seconds per particle. Defaults to (0.005, 0.02).
        """
        self.fitness_f = fitness_f
        self.latency_range = latency_range

    async def eval_one(self, position: np.ndarray) -> float:
        """
        Evaluate one particle asynchronously, simulating I/O latency.

        The await suspends this coroutine without blocking the event loop,
        allowing other particle evaluations to proceed concurrently.

        Args:
            position (np.ndarray): Position of one particle.

        Returns:
            float: Fitness value for that particle.
        """
        latency = random.uniform(*self.latency_range)
        await asyncio.sleep(latency)   # yields control to the event loop
        return self.fitness_f(position)

    async def gather_all(self, positions: np.ndarray) -> list[float]:
        """
        Launch all particle evaluations concurrently and wait for all results.

        asyncio.gather() schedules all coroutines at once. Each one runs until
        it hits its await asyncio.sleep(), suspends, and lets the others start.
        They all resume roughly at the same time when their latency expires.

        Args:
            positions (np.ndarray): Position matrix of shape (n_particles, dim).

        Returns:
            list[float]: Fitness values in the same order as positions.
        """
        coroutines = [self.eval_one(pos) for pos in positions]
        return await asyncio.gather(*coroutines)

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitness of all particles using async concurrency.

        asyncio.run() creates a new event loop, runs _gather_all() to completion,
        and returns the results. This bridges the synchronous PSO engine with
        the async evaluation logic.

        Args:
            positions (np.ndarray): Position matrix of shape (n_particles, dim).

        Returns:
            np.ndarray: Fitness values of shape (n_particles,).
        """
        values = asyncio.run(self.gather_all(positions))
        return np.array(values, dtype=float)

    def shutdown(self) -> None:
        """Release evaluator resources. No-op for the async evaluator."""
        pass