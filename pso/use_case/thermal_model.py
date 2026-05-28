# @author: Íñigo Martínez Jiménez
# This module defines the simplified thermal model used by the data center
# cooling use case, including the effective cooling per rack and the static
# rack temperature simulation

import numpy as np

from pso.use_case.scenario import DataCenterScenario
from pso.use_case.encoding import decode_particle


def compute_effective_cooling(x_norm: np.ndarray, scenario: DataCenterScenario) -> np.ndarray:
    """
    Compute the effective cooling F_r received by each rack from fans and airflow.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the influence matrices.

    Returns:
        np.ndarray: Effective cooling values of shape (n_racks,).
    """
    _, fans, zones = decode_particle(x_norm, scenario)
    return (
        scenario.f_min
        + scenario.fan_influence_matrix @ fans
        + scenario.zone_influence_matrix @ zones
    )


def simulate_temperatures(x_norm: np.ndarray, scenario: DataCenterScenario) -> np.ndarray:
    """
    Simulate the rack temperatures for a given configuration using a static
    thermal model that combines the inlet setpoint, self-heating, neighbour
    coupling, and a fixed per-rack bias.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding loads, coupling and biases.

    Returns:
        np.ndarray: Simulated rack temperatures of shape (n_racks,).
    """
    t_set, fans, zones = decode_particle(x_norm, scenario)

    F = (
        scenario.f_min
        + scenario.fan_influence_matrix @ fans
        + scenario.zone_influence_matrix @ zones
    )

    # Heat dissipated by each rack with its own cooling
    own_term = scenario.alpha * scenario.rack_loads / (F + scenario.epsilon)

    # Extra heat received from neighbours weighted by the coupling matrix
    cross_load = scenario.rack_loads / (F + scenario.epsilon)
    coupled_term = scenario.beta * (scenario.thermal_coupling_matrix @ cross_load)

    return t_set + own_term + coupled_term + scenario.thermal_bias
