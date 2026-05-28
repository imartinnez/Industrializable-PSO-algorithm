# @author: Íñigo Martínez Jiménez
# This module defines the five soft penalties that the data center cooling
# objective applies on top of the normalised energy term: safety, hotspot,
# overcooling, thermal balance, and change from the baseline configuration

import numpy as np

from pso.use_case.scenario import DataCenterScenario
from pso.use_case.thermal_model import simulate_temperatures


def compute_penalties(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Compute the five soft penalties used by the fitness function.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the limits and baseline.

    Returns:
        dict: Dictionary with the five penalty values.
    """
    T = simulate_temperatures(x_norm, scenario)

    # Average squared excess over the soft safety limit
    safe_excess = np.maximum(0.0, T - scenario.t_safe)
    safe_penalty = float(np.mean(safe_excess ** 2) / scenario.delta_t ** 2)

    # Squared excess of the hottest rack over the hard hotspot limit
    hot_excess = max(0.0, float(np.max(T)) - scenario.t_hot)
    hot_penalty = float(hot_excess ** 2 / scenario.delta_t_hot ** 2)

    # Average squared shortfall below the minimum useful temperature
    over_excess = np.maximum(0.0, scenario.t_min - T)
    over_penalty = float(np.mean(over_excess ** 2) / scenario.delta_t ** 2)

    # Variance across racks normalised by the temperature scale
    balance_penalty = float((np.std(T) / scenario.delta_t) ** 2)

    # Squared distance to the baseline configuration in normalised units
    x_phys = scenario.lower_phys + x_norm * (scenario.upper_phys - scenario.lower_phys)
    diffs = (x_phys - scenario.baseline_x_phys) / (scenario.upper_phys - scenario.lower_phys)
    change_penalty = float(np.mean(diffs ** 2))

    return {
        "safe_penalty": safe_penalty,
        "hotspot_penalty": hot_penalty,
        "overcooling_penalty": over_penalty,
        "balance_penalty": balance_penalty,
        "change_penalty": change_penalty,
    }
