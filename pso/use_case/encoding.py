# @author: Íñigo Martínez Jiménez
# This module defines the two helpers that translate particles between the
# normalised space used by the PSO and the physical units used by the
# data center cooling model

import numpy as np

from pso.use_case.scenario import DataCenterScenario


def phys_to_norm(x_phys: np.ndarray, scenario: DataCenterScenario) -> np.ndarray:
    """
    Map a physical configuration to its normalised representation in [0, 1].

    Args:
        x_phys (np.ndarray): Configuration expressed in physical units.
        scenario (DataCenterScenario): Scenario holding the lower and upper bounds.

    Returns:
        np.ndarray: Normalised configuration with values in [0, 1].
    """
    return (x_phys - scenario.lower_phys) / (scenario.upper_phys - scenario.lower_phys)


def decode_particle(x_norm: np.ndarray, scenario: DataCenterScenario) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Decode a normalised particle into the three physical components used by the
    thermal and energy models.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the lower and upper bounds.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Setpoint temperature, fan speeds and zone airflows.
    """
    x_phys = scenario.lower_phys + x_norm * (scenario.upper_phys - scenario.lower_phys)
    t_set = float(x_phys[0])
    fan_speeds = x_phys[1:1 + scenario.n_fans]
    zone_airflows = x_phys[1 + scenario.n_fans:]
    return t_set, fan_speeds, zone_airflows