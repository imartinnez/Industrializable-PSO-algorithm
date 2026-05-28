# @author: Íñigo Martínez Jiménez
# This module defines the penalised fitness function used by the PSO on the
# data center cooling use case, the closure factory that adapts it to the PSO
# engine, and a rich evaluation helper used for reporting and plotting

from collections.abc import Callable
import numpy as np

from pso.use_case.scenario import DataCenterScenario
from pso.use_case.encoding import decode_particle
from pso.use_case.thermal_model import simulate_temperatures
from pso.use_case.energy_model import compute_energy
from pso.use_case.penalties import compute_penalties


class DataCenterObjective:
    """
    Bind the data center cooling fitness with its scenario so it can be passed
    to the PSO engine as a plain callable through the evaluate method.
    """

    def __init__(self, scenario: DataCenterScenario) -> None:
        """
        Store the scenario used by every fitness evaluation.

        Args:
            scenario (DataCenterScenario): Scenario used by the bound objective.
        """
        self.scenario = scenario

    def evaluate(self, x_norm: np.ndarray) -> float:
        """
        Evaluate the penalised fitness on a normalised particle.

        Args:
            x_norm (np.ndarray): Particle position in [0, 1]^dim.

        Returns:
            float: Penalised fitness value.
        """
        return datacenter_cooling_objective(x_norm, self.scenario)





def datacenter_cooling_objective(x_norm: np.ndarray, scenario: DataCenterScenario) -> float:
    """
    Compute the full penalised fitness used by the PSO. Lower is better.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding all model parameters.

    Returns:
        float: Fitness combining normalised energy and the five soft penalties.
    """
    energy = compute_energy(x_norm, scenario)
    penalties = compute_penalties(x_norm, scenario)

    fitness = (energy["total_energy"] / scenario.baseline_energy + scenario.lambda_safe * penalties["safe_penalty"] + scenario.lambda_hot * penalties["hotspot_penalty"] + scenario.lambda_over * penalties["overcooling_penalty"] + scenario.lambda_balance * penalties["balance_penalty"] + scenario.lambda_change * penalties["change_penalty"])
    return float(fitness)

def make_objective(scenario: DataCenterScenario) -> Callable[[np.ndarray], float]:
    """
    Build a fitness function compatible with the PSO engine by binding the
    scenario to a DataCenterObjective and returning its evaluate method.

    Args:
        scenario (DataCenterScenario): Scenario used by the bound objective.

    Returns:
        Callable[[np.ndarray], float]: Method ready to be passed as fitness_f.
    """
    return DataCenterObjective(scenario).evaluate



def evaluate_solution(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Run the full evaluation pipeline on one configuration and return a rich
    dictionary with the information needed for reporting, plotting and persistence.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding all model parameters.

    Returns:
        dict: Dictionary with fitness, energy breakdown, temperature statistics,
            safety counts, decoded variables and the raw temperature array.
    """
    t_set, fan_speeds, zone_airflows = decode_particle(x_norm, scenario)
    T = simulate_temperatures(x_norm, scenario)
    energy = compute_energy(x_norm, scenario)
    penalties = compute_penalties(x_norm, scenario)
    fitness = datacenter_cooling_objective(x_norm, scenario)

    return {
        "fitness": fitness,
        "total_energy": energy["total_energy"],
        "chiller_energy": energy["chiller_energy"],
        "fan_energy": energy["fan_energy"],
        "airflow_energy": energy["airflow_energy"],
        "mean_temperature": float(np.mean(T)),
        "max_temperature": float(np.max(T)),
        "min_temperature": float(np.min(T)),
        "std_temperature": float(np.std(T)),
        "unsafe_racks": int(np.sum(T > scenario.t_safe)),
        "hotspots": int(np.sum(T > scenario.t_hot)),
        "overcooled_racks": int(np.sum(T < scenario.t_min)),
        "temperatures": T,
        "penalties": penalties,
        "t_set": t_set,
        "fan_speeds": fan_speeds,
        "zone_airflows": zone_airflows,
    }
