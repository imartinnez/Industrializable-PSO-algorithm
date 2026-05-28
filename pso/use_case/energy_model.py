# @author: Íñigo Martínez Jiménez
# This module defines the energy model of the data center cooling use case,
# including the chiller, fan and airflow contributions and the total energy

import numpy as np

from pso.use_case.scenario import DataCenterScenario
from pso.use_case.encoding import decode_particle


def compute_energy(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Compute the energy consumption of a configuration, broken down into the
    chiller, the fans, and the airflow contributions.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the energy parameters.

    Returns:
        dict: Dictionary with chiller_energy, fan_energy, airflow_energy and total_energy.
    """
    t_set, fans, zones = decode_particle(x_norm, scenario)

    # Chiller cost grows exponentially when the setpoint moves below the reference
    e_chiller = scenario.p_chiller_ref * np.exp(scenario.gamma * (scenario.t_ref - t_set))

    # Fans and airflow follow the standard cubic law on the actuator effort
    e_fans = scenario.p_fan_max * np.sum(fans ** 3)
    e_airflow = scenario.p_flow_max * np.sum(zones ** 3)

    return {"chiller_energy": float(e_chiller), "fan_energy": float(e_fans), "airflow_energy": float(e_airflow), "total_energy": float(e_chiller + e_fans + e_airflow)}
