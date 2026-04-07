# @author: Íñigo Martínez Jiménez
# This module defines the unit tests for the PSO implementation,
# covering reproducibility, bounds handling, monotonic convergence,
# basic correctness on Sphere, and result consistency

import numpy as np

import pso.objectives.registry as o
from pso.experiments.benchmarks import Instance


sphere = o.get_objective("sphere")

# Base configuration used across the tests
BASE = dict(
    fitness_f=sphere.function,
    dim=5,
    constraints=sphere.constraints,
    seed=42,
    max_iter=500,
    patience=100,
    imp_min=1e-8,
    n_particles=30,
    strategy="clamp",
    fitness_policy="plain",
    topology="global",
    tol=0.0,
    mode="sequential",
    optimum_value=sphere.optimum_value,
)


def make_instance(**overrides) -> Instance:
    """
    Create a test instance starting from a shared base configuration.

    Args:
        **overrides: Parameters that replace the default test configuration.

    Returns:
        Instance: Configured PSO experiment instance.
    """
    params = {**BASE, **overrides}
    return Instance(name="test", **params)


# Reproducibility

def test_same_seed_same_result() -> None:
    """
    Check that using the same seed produces the same final result.
    """
    r1 = make_instance(seed=1).run_instance()
    r2 = make_instance(seed=1).run_instance()

    assert r1.b_value == r2.b_value
    assert np.array_equal(r1.b_position, r2.b_position)


def test_different_seeds_different_results() -> None:
    """
    Check that different seeds can produce different final positions.
    """
    r1 = make_instance(seed=1).run_instance()
    r2 = make_instance(seed=2).run_instance()

    assert not np.array_equal(r1.b_position, r2.b_position)


# Bounds

def test_best_position_within_bounds() -> None:
    """
    Check that the best position found stays inside the search bounds.
    """
    result = make_instance().run_instance()
    low, high = sphere.constraints

    assert np.all(result.b_position >= low)
    assert np.all(result.b_position <= high)


def test_bounds_respected_across_dims() -> None:
    """
    Check that the best position respects the bounds for different dimensions.
    """
    for dim in [2, 5, 10]:
        result = make_instance(dim=dim).run_instance()
        low, high = sphere.constraints

        assert result.b_position.shape == (dim,)
        assert np.all(result.b_position >= low)
        assert np.all(result.b_position <= high)


# Monotonicity

def test_global_best_never_worsens() -> None:
    """
    Check that the global best fitness never gets worse during the run.
    """
    result = make_instance().run_instance()
    curve = result.best_fitness_by_iter

    for i in range(1, len(curve)):
        assert curve[i] <= curve[i - 1] + 1e-12


# Correctness on Sphere

def test_sphere_converges_to_zero() -> None:
    """
    Check that PSO converges close to zero on the Sphere function.
    """
    result = make_instance(dim=3, max_iter=1000, n_particles=50).run_instance()

    assert result.b_value < 1e-4


def test_sphere_best_value_non_negative() -> None:
    """
    Check that the best value found on Sphere is never negative.
    """
    result = make_instance().run_instance()

    assert result.b_value >= 0.0


# Result structure

def test_iterations_within_limit() -> None:
    """
    Check that the number of executed iterations stays within the allowed limit.
    """
    result = make_instance(max_iter=200).run_instance()

    assert 1 <= result.iterations <= 200


def test_result_fields_consistent() -> None:
    """
    Check that the main result fields are internally consistent.
    """
    result = make_instance().run_instance()

    assert len(result.best_fitness_by_iter) == result.iterations
    assert result.fitness_eval_time_total <= result.total_time
    assert result.total_time > 0