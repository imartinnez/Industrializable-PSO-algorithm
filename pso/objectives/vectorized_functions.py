# @author: Íñigo Martínez Jiménez
# This module defines vectorized versions of the benchmark objective functions.
# Unlike the scalar versions in functions.py, these accept a full position matrix
# of shape (n_particles, dim) and return a fitness array of shape (n_particles,),
# computing all particle evaluations in a single NumPy operation.

import numpy as np


def sphere_vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorized Sphere function over all particles.

    Args:
        X (np.ndarray): Position matrix of shape (n_particles, dim).

    Returns:
        np.ndarray: Fitness values of shape (n_particles,).
    """
    return np.sum(X ** 2, axis=1)


def rosenbrock_vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorized Rosenbrock function over all particles.

    Args:
        X (np.ndarray): Position matrix of shape (n_particles, dim).

    Returns:
        np.ndarray: Fitness values of shape (n_particles,).
    """
    # X[:, :-1] are all dimensions except the last, X[:, 1:] are all except the first
    return np.sum(
        100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2,
        axis=1
    )


def rastrigin_vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorized Rastrigin function over all particles.

    Args:
        X (np.ndarray): Position matrix of shape (n_particles, dim).

    Returns:
        np.ndarray: Fitness values of shape (n_particles,).
    """
    d = X.shape[1]
    return 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley_vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorized Ackley function over all particles.

    Args:
        X (np.ndarray): Position matrix of shape (n_particles, dim).

    Returns:
        np.ndarray: Fitness values of shape (n_particles,).
    """
    d = X.shape[1]
    sum_sq  = np.sum(X ** 2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * X), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e