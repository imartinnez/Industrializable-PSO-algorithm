# @author: Íñigo Martínez Jiménez
# This module defines the unit tests for the data center cooling use case,
# covering scenario reproducibility, bounds and dimensions, particle decoding,
# thermal and energy model behaviour, safety penalties, and the structure of
# the evaluate_solution output

import numpy as np
import pytest

from pso.use_case.datacenter_cooling import (
    create_default_datacenter_scenario,
    decode_particle,
    compute_effective_cooling,
    simulate_temperatures,
    compute_energy,
    datacenter_cooling_objective,
    evaluate_solution,
    make_objective,
    phys_to_norm,
)


# Reproducibility

def test_scenario_is_reproducible() -> None:
    """
    Check that two scenarios built with the same seed share the same random parts.
    """
    s1 = create_default_datacenter_scenario(seed=123)
    s2 = create_default_datacenter_scenario(seed=123)

    assert np.array_equal(s1.rack_loads, s2.rack_loads)
    assert np.array_equal(s1.thermal_bias, s2.thermal_bias)
    assert s1.baseline_energy == s2.baseline_energy


def test_different_seeds_produce_different_scenarios() -> None:
    """
    Check that different seeds produce different rack loads.
    """
    s1 = create_default_datacenter_scenario(seed=1)
    s2 = create_default_datacenter_scenario(seed=2)

    assert not np.array_equal(s1.rack_loads, s2.rack_loads)


# Bounds and dimensions

def test_dim_matches_layout() -> None:
    """
    Check that the PSO dimension equals 1 + n_fans + n_zones and the bounds match.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    expected = 1 + scenario.n_fans + scenario.n_zones

    assert scenario.dim == expected
    assert scenario.lower_phys.shape == (expected,)
    assert scenario.upper_phys.shape == (expected,)
    assert scenario.baseline_x_phys.shape == (expected,)


# Decoding

def test_decode_particle_separates_components() -> None:
    """
    Check that decoding a zero vector returns all the lower physical bounds.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.zeros(scenario.dim)
    t_set, fans, zones = decode_particle(x, scenario)

    assert t_set == pytest.approx(scenario.lower_phys[0])
    assert fans.shape == (scenario.n_fans,)
    assert zones.shape == (scenario.n_zones,)
    np.testing.assert_allclose(fans, scenario.lower_phys[1:1 + scenario.n_fans])
    np.testing.assert_allclose(zones, scenario.lower_phys[1 + scenario.n_fans:])


def test_decode_uniform_one_yields_upper_bounds() -> None:
    """
    Check that decoding a vector of ones returns all the upper physical bounds.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.ones(scenario.dim)
    t_set, fans, zones = decode_particle(x, scenario)

    assert t_set == pytest.approx(scenario.upper_phys[0])
    np.testing.assert_allclose(fans, scenario.upper_phys[1:1 + scenario.n_fans])
    np.testing.assert_allclose(zones, scenario.upper_phys[1 + scenario.n_fans:])


# Simulation outputs

def test_simulate_temperatures_shape() -> None:
    """
    Check that simulate_temperatures returns a finite array of length n_racks.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.full(scenario.dim, 0.5)
    T = simulate_temperatures(x, scenario)

    assert T.shape == (scenario.n_racks,)
    assert np.all(np.isfinite(T))


def test_effective_cooling_positive() -> None:
    """
    Check that the effective cooling is strictly positive for any valid input.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.full(scenario.dim, 0.5)
    F = compute_effective_cooling(x, scenario)

    assert np.all(F > 0.0)


# Energy model

def test_energy_components_positive() -> None:
    """
    Check that all energy components return positive values for any valid input.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.full(scenario.dim, 0.5)
    e = compute_energy(x, scenario)

    assert e["chiller_energy"] > 0
    assert e["fan_energy"] > 0
    assert e["airflow_energy"] > 0
    assert e["total_energy"] > 0


def test_fan_energy_grows_with_fan_speed() -> None:
    """
    Check that increasing the fan speeds increases the fan energy term.
    """
    scenario = create_default_datacenter_scenario(seed=42)

    x_low = np.full(scenario.dim, 0.5)
    x_low[1:1 + scenario.n_fans] = 0.1

    x_high = np.full(scenario.dim, 0.5)
    x_high[1:1 + scenario.n_fans] = 0.9

    e_low = compute_energy(x_low, scenario)
    e_high = compute_energy(x_high, scenario)
    assert e_high["fan_energy"] > e_low["fan_energy"]


def test_chiller_energy_grows_when_setpoint_drops() -> None:
    """
    Check that lowering the temperature setpoint increases the chiller energy term.
    """
    scenario = create_default_datacenter_scenario(seed=42)

    x_warm = np.full(scenario.dim, 0.5)
    x_warm[0] = 0.9

    x_cold = np.full(scenario.dim, 0.5)
    x_cold[0] = 0.1

    e_warm = compute_energy(x_warm, scenario)
    e_cold = compute_energy(x_cold, scenario)
    assert e_cold["chiller_energy"] > e_warm["chiller_energy"]


# Objective

def test_objective_is_finite_float() -> None:
    """
    Check that the fitness is a finite float for any valid input.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.full(scenario.dim, 0.5)
    f = datacenter_cooling_objective(x, scenario)

    assert isinstance(f, float)
    assert np.isfinite(f)


def test_unsafe_configuration_has_worse_fitness() -> None:
    """
    Check that an unsafe configuration with high T_set and minimum cooling
    receives a worse fitness than the baseline configuration.
    """
    scenario = create_default_datacenter_scenario(seed=42)

    x_baseline_norm = phys_to_norm(scenario.baseline_x_phys, scenario)
    f_baseline = datacenter_cooling_objective(x_baseline_norm, scenario)

    # High setpoint plus minimum fans and zones leaves the racks under-cooled
    x_unsafe = np.zeros(scenario.dim)
    x_unsafe[0] = 1.0
    f_unsafe = datacenter_cooling_objective(x_unsafe, scenario)

    assert f_unsafe > f_baseline


# Evaluate_solution

def test_evaluate_solution_schema() -> None:
    """
    Check that evaluate_solution returns all the metrics expected by the run script.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    x = np.full(scenario.dim, 0.5)
    result = evaluate_solution(x, scenario)

    expected_keys = {
        "fitness", "total_energy", "chiller_energy", "fan_energy", "airflow_energy",
        "mean_temperature", "max_temperature", "min_temperature", "std_temperature",
        "unsafe_racks", "hotspots", "overcooled_racks", "temperatures",
        "penalties", "t_set", "fan_speeds", "zone_airflows",
    }
    assert expected_keys.issubset(result.keys())
    assert result["temperatures"].shape == (scenario.n_racks,)
    assert set(result["penalties"].keys()) == {
        "safe_penalty", "hotspot_penalty", "overcooling_penalty",
        "balance_penalty", "change_penalty",
    }


# Closure for PSO

def test_make_objective_returns_callable_float() -> None:
    """
    Check that the closure returned by make_objective is callable and finite.
    """
    scenario = create_default_datacenter_scenario(seed=42)
    obj = make_objective(scenario)
    x = np.full(scenario.dim, 0.5)
    value = obj(x)

    assert isinstance(value, float)
    assert np.isfinite(value)
