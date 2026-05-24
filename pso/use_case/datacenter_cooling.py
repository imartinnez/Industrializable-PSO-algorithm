# @author: Íñigo Martínez Jiménez
# This module defines the data center cooling use case used by the PSO optimizer.
# It contains the DataCenterScenario dataclass, the thermal and energy models,
# the penalised objective function, and small helpers to decode, simulate and
# evaluate a candidate configuration.

from dataclasses import dataclass
from collections.abc import Callable
import numpy as np


# Default scenario constants

# Layout
N_FANS = 4
N_ZONES = 4
GRID_SHAPE = (4, 5)

# Decoded variable bounds
T_SET_LOW, T_SET_HIGH = 18.0, 26.0
FAN_LOW, FAN_HIGH = 0.3, 1.0
ZONE_LOW, ZONE_HIGH = 0.2, 1.0

# Thermal model parameters
T_MIN = 18.0
T_SAFE = 27.0
T_HOT = 30.0
T_REF = 22.0
ALPHA = 1.5
BETA = 0.2
EPSILON = 0.1
F_MIN = 0.2
SIGMA_FAN = 2.0
SIGMA_THERMAL = 1.5

# Energy model parameters
P_CHILLER_REF = 40.0
P_FAN_MAX = 5.0
P_FLOW_MAX = 3.0
GAMMA = 0.08

# Penalty weights and scaling
LAMBDA_SAFE = 50.0
LAMBDA_HOT = 100.0
LAMBDA_OVER = 5.0
LAMBDA_BALANCE = 2.0
LAMBDA_CHANGE = 1.0
DELTA_T = 1.0
DELTA_T_HOT = 1.0


@dataclass
class DataCenterScenario:
    """
    Container with the layout, thermal coupling and energy parameters used
    by the data center cooling use case. All matrices and vectors are
    pre-built so that the objective function only does cheap arithmetic.
    """
    n_racks: int
    n_fans: int
    n_zones: int
    grid_shape: tuple[int, int]

    rack_positions: np.ndarray            # (n_racks, 2)
    fan_positions: np.ndarray             # (n_fans, 2)
    rack_loads: np.ndarray                # (n_racks,)
    fan_influence_matrix: np.ndarray      # A: (n_racks, n_fans)
    zone_influence_matrix: np.ndarray     # B: (n_racks, n_zones)
    thermal_coupling_matrix: np.ndarray   # C: (n_racks, n_racks)
    thermal_bias: np.ndarray              # eta: (n_racks,)

    p_chiller_ref: float
    p_fan_max: float
    p_flow_max: float
    gamma: float
    t_ref: float

    alpha: float
    beta: float
    epsilon: float
    f_min: float

    t_min: float
    t_safe: float
    t_hot: float
    delta_t: float
    delta_t_hot: float

    lambda_safe: float
    lambda_hot: float
    lambda_over: float
    lambda_balance: float
    lambda_change: float

    baseline_x_phys: np.ndarray           # (dim,)
    lower_phys: np.ndarray                # (dim,)
    upper_phys: np.ndarray                # (dim,)
    baseline_energy: float

    @property
    def dim(self) -> int:
        """Dimension of the PSO search space: T_set + fans + zones."""
        return 1 + self.n_fans + self.n_zones


# Internal builders used to construct the default scenario

def _build_rack_positions(grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Build the (row, col) integer positions of all racks in row-major order.

    Args:
        grid_shape (tuple[int, int]): Number of rows and columns of the rack grid.

    Returns:
        np.ndarray: Array of shape (n_racks, 2) with rack positions.
    """
    rows, cols = grid_shape
    positions = np.array(
        [[i, j] for i in range(rows) for j in range(cols)],
        dtype=float,
    )
    return positions


def _build_fan_positions(grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Place the four fans on the corners of the rack grid bounding box.

    Args:
        grid_shape (tuple[int, int]): Number of rows and columns of the rack grid.

    Returns:
        np.ndarray: Array of shape (4, 2) with fan positions.
    """
    rows, cols = grid_shape
    return np.array(
        [
            [0, 0],
            [0, cols - 1],
            [rows - 1, 0],
            [rows - 1, cols - 1],
        ],
        dtype=float,
    )


def _build_fan_influence_matrix(rack_positions: np.ndarray, fan_positions: np.ndarray,
                                 sigma: float) -> np.ndarray:
    """
    Build the fan-to-rack influence matrix A using a Gaussian decay with distance,
    and normalise each column so the cooling delivered by each fan sums to one.

    Args:
        rack_positions (np.ndarray): Rack positions of shape (n_racks, 2).
        fan_positions (np.ndarray): Fan positions of shape (n_fans, 2).
        sigma (float): Spatial scale of the Gaussian decay.

    Returns:
        np.ndarray: Influence matrix of shape (n_racks, n_fans).
    """
    n_racks = len(rack_positions)
    n_fans = len(fan_positions)
    A = np.zeros((n_racks, n_fans), dtype=float)

    for j in range(n_fans):
        d2 = np.sum((rack_positions - fan_positions[j]) ** 2, axis=1)
        A[:, j] = np.exp(-d2 / (2 * sigma ** 2))

    # Normalise each fan column: the total cooling output of one fan sums to 1
    A = A / A.sum(axis=0, keepdims=True)
    return A


def _build_zone_influence_matrix(rack_positions: np.ndarray, grid_shape: tuple[int, int],
                                  n_zones: int) -> np.ndarray:
    """
    Assign each rack to one of the four grid quadrants and build a one-hot matrix.

    Args:
        rack_positions (np.ndarray): Rack positions of shape (n_racks, 2).
        grid_shape (tuple[int, int]): Number of rows and columns of the rack grid.
        n_zones (int): Number of zones (must be 4 in the default scenario).

    Returns:
        np.ndarray: Indicator matrix of shape (n_racks, n_zones).
    """
    rows, cols = grid_shape
    n_racks = len(rack_positions)
    B = np.zeros((n_racks, n_zones), dtype=float)

    row_half = rows / 2
    col_half = cols / 2

    for r in range(n_racks):
        ri, rj = rack_positions[r, 0], rack_positions[r, 1]
        top = ri < row_half
        left = rj < col_half
        if top and left:
            z = 0
        elif top and not left:
            z = 1
        elif not top and left:
            z = 2
        else:
            z = 3
        B[r, z] = 1.0

    return B


def _build_thermal_coupling_matrix(rack_positions: np.ndarray, sigma: float) -> np.ndarray:
    """
    Build a symmetric thermal coupling matrix C using a Gaussian decay with distance,
    with zero diagonal so a rack does not couple with itself in the cross term.

    Args:
        rack_positions (np.ndarray): Rack positions of shape (n_racks, 2).
        sigma (float): Spatial scale of the Gaussian decay.

    Returns:
        np.ndarray: Coupling matrix of shape (n_racks, n_racks).
    """
    n_racks = len(rack_positions)
    C = np.zeros((n_racks, n_racks), dtype=float)
    for r in range(n_racks):
        d2 = np.sum((rack_positions - rack_positions[r]) ** 2, axis=1)
        C[r] = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(C, 0.0)
    return C


def create_default_datacenter_scenario(seed: int = 42) -> DataCenterScenario:
    """
    Build a reproducible default data center scenario using a fixed seed for
    the random rack loads and thermal biases.

    Args:
        seed (int): Random seed used to make the scenario reproducible.

    Returns:
        DataCenterScenario: Fully built scenario with the baseline energy precomputed.
    """
    rng = np.random.default_rng(seed)

    grid_shape = GRID_SHAPE
    n_racks = grid_shape[0] * grid_shape[1]
    n_fans = N_FANS
    n_zones = N_ZONES

    rack_positions = _build_rack_positions(grid_shape)
    fan_positions = _build_fan_positions(grid_shape)

    rack_loads = rng.uniform(0.7, 1.3, size=n_racks)
    thermal_bias = rng.normal(0.0, 0.3, size=n_racks)

    A = _build_fan_influence_matrix(rack_positions, fan_positions, SIGMA_FAN)
    B = _build_zone_influence_matrix(rack_positions, grid_shape, n_zones)
    C = _build_thermal_coupling_matrix(rack_positions, SIGMA_THERMAL)

    # Baseline configuration expressed in physical units
    baseline_x_phys = np.concatenate(
        [
            [22.0],
            [0.75] * n_fans,
            [0.70] * n_zones,
        ]
    )

    # Physical bounds aligned with baseline_x_phys order
    lower_phys = np.concatenate(
        [
            [T_SET_LOW],
            [FAN_LOW] * n_fans,
            [ZONE_LOW] * n_zones,
        ]
    )
    upper_phys = np.concatenate(
        [
            [T_SET_HIGH],
            [FAN_HIGH] * n_fans,
            [ZONE_HIGH] * n_zones,
        ]
    )

    # Build the scenario with a placeholder baseline_energy, then fill it in
    scenario = DataCenterScenario(
        n_racks=n_racks,
        n_fans=n_fans,
        n_zones=n_zones,
        grid_shape=grid_shape,
        rack_positions=rack_positions,
        fan_positions=fan_positions,
        rack_loads=rack_loads,
        fan_influence_matrix=A,
        zone_influence_matrix=B,
        thermal_coupling_matrix=C,
        thermal_bias=thermal_bias,
        p_chiller_ref=P_CHILLER_REF,
        p_fan_max=P_FAN_MAX,
        p_flow_max=P_FLOW_MAX,
        gamma=GAMMA,
        t_ref=T_REF,
        alpha=ALPHA,
        beta=BETA,
        epsilon=EPSILON,
        f_min=F_MIN,
        t_min=T_MIN,
        t_safe=T_SAFE,
        t_hot=T_HOT,
        delta_t=DELTA_T,
        delta_t_hot=DELTA_T_HOT,
        lambda_safe=LAMBDA_SAFE,
        lambda_hot=LAMBDA_HOT,
        lambda_over=LAMBDA_OVER,
        lambda_balance=LAMBDA_BALANCE,
        lambda_change=LAMBDA_CHANGE,
        baseline_x_phys=baseline_x_phys,
        lower_phys=lower_phys,
        upper_phys=upper_phys,
        baseline_energy=0.0,
    )

    # Compute the actual baseline energy and store it in the scenario
    x_baseline_norm = phys_to_norm(baseline_x_phys, scenario)
    scenario.baseline_energy = compute_energy(x_baseline_norm, scenario)["total_energy"]
    return scenario


# Encoding / decoding between normalised and physical variables

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
    Decode a normalised particle into its three physical components.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the lower and upper bounds.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: Setpoint temperature, fan speeds,
            and zone airflows.
    """
    x_phys = scenario.lower_phys + x_norm * (scenario.upper_phys - scenario.lower_phys)
    t_set = float(x_phys[0])
    fan_speeds = x_phys[1:1 + scenario.n_fans]
    zone_airflows = x_phys[1 + scenario.n_fans:]
    return t_set, fan_speeds, zone_airflows


# Thermal and energy models

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
    F = (
        scenario.f_min
        + scenario.fan_influence_matrix @ fans
        + scenario.zone_influence_matrix @ zones
    )
    return F


def simulate_temperatures(x_norm: np.ndarray, scenario: DataCenterScenario) -> np.ndarray:
    """
    Simulate the rack temperatures for a given configuration using a simplified
    static thermal model with rack self-heating and neighbour coupling.

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

    # Per-rack heat removed by its own cooling
    own_term = scenario.alpha * scenario.rack_loads / (F + scenario.epsilon)

    # Cross term that captures heat received from neighbouring racks
    cross_load = scenario.rack_loads / (F + scenario.epsilon)
    coupled_term = scenario.beta * (scenario.thermal_coupling_matrix @ cross_load)

    return t_set + own_term + coupled_term + scenario.thermal_bias


def compute_energy(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Compute the energy consumption of a configuration, broken down into chiller,
    fans, and airflow contributions.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the energy parameters.

    Returns:
        dict: Dictionary with chiller_energy, fan_energy, airflow_energy and total_energy.
    """
    t_set, fans, zones = decode_particle(x_norm, scenario)

    e_chiller = scenario.p_chiller_ref * np.exp(scenario.gamma * (scenario.t_ref - t_set))
    e_fans = scenario.p_fan_max * np.sum(fans ** 3)
    e_airflow = scenario.p_flow_max * np.sum(zones ** 3)

    return {
        "chiller_energy": float(e_chiller),
        "fan_energy": float(e_fans),
        "airflow_energy": float(e_airflow),
        "total_energy": float(e_chiller + e_fans + e_airflow),
    }


def compute_penalties(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Compute the five soft penalties used by the fitness function:
    safety, hotspot, overcooling, thermal balance, and change from baseline.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding the limits and baseline.

    Returns:
        dict: Dictionary with the five penalty values.
    """
    T = simulate_temperatures(x_norm, scenario)

    safe_excess = np.maximum(0.0, T - scenario.t_safe)
    safe_penalty = float(np.mean(safe_excess ** 2) / scenario.delta_t ** 2)

    hot_excess = max(0.0, float(np.max(T)) - scenario.t_hot)
    hot_penalty = float(hot_excess ** 2 / scenario.delta_t_hot ** 2)

    over_excess = np.maximum(0.0, scenario.t_min - T)
    over_penalty = float(np.mean(over_excess ** 2) / scenario.delta_t ** 2)

    balance_penalty = float((np.std(T) / scenario.delta_t) ** 2)

    # Change penalty uses physical values normalised by the physical range
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


def datacenter_cooling_objective(x_norm: np.ndarray, scenario: DataCenterScenario) -> float:
    """
    Compute the full penalised fitness used by the PSO. Lower is better.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding all model parameters.

    Returns:
        float: Fitness value combining normalised energy and the five penalties.
    """
    energy = compute_energy(x_norm, scenario)
    penalties = compute_penalties(x_norm, scenario)

    fitness = (
        energy["total_energy"] / scenario.baseline_energy
        + scenario.lambda_safe * penalties["safe_penalty"]
        + scenario.lambda_hot * penalties["hotspot_penalty"]
        + scenario.lambda_over * penalties["overcooling_penalty"]
        + scenario.lambda_balance * penalties["balance_penalty"]
        + scenario.lambda_change * penalties["change_penalty"]
    )
    return float(fitness)


def evaluate_solution(x_norm: np.ndarray, scenario: DataCenterScenario) -> dict:
    """
    Run the full evaluation pipeline on one configuration and return a rich
    dictionary suitable for reporting, plotting and persistence.

    Args:
        x_norm (np.ndarray): Particle position in [0, 1]^dim.
        scenario (DataCenterScenario): Scenario holding all model parameters.

    Returns:
        dict: Dictionary with fitness, energy breakdown, temperature statistics,
            safety counts, decoded variables, and the raw temperature array.
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


def make_objective(scenario: DataCenterScenario) -> Callable[[np.ndarray], float]:
    """
    Build a fitness function compatible with the PSO engine by closing over
    the scenario, so the engine can call it as f(x) -> float.

    Args:
        scenario (DataCenterScenario): Scenario used by the closed-over objective.

    Returns:
        Callable[[np.ndarray], float]: Function ready to be passed as fitness_f.
    """
    def objective(x_norm: np.ndarray) -> float:
        return datacenter_cooling_objective(x_norm, scenario)
    return objective
