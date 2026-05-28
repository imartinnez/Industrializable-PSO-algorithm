# @author: Íñigo Martínez Jiménez
# This module defines the DataCenterScenario dataclass and the factory that
# builds the default scenario used by the data center cooling use case,
# including all the layout, thermal coupling and energy parameters

from dataclasses import dataclass
import numpy as np


# Default scenario constants used by create_default_datacenter_scenario

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
    Store the full configuration of the data center used by the cooling use case.
    All matrices and vectors are pre-built so the objective function only does
    cheap arithmetic at evaluation time.

    Args:
        n_racks (int): Number of racks in the data center.
        n_fans (int): Number of fans.
        n_zones (int): Number of airflow zones.
        dim (int): PSO search dimension, equal to 1 + n_fans + n_zones.
        grid_shape (tuple[int, int]): Layout of the rack grid as (rows, cols).
        rack_positions (np.ndarray): Rack positions of shape (n_racks, 2) with integer-valued row and column indices.
        fan_positions (np.ndarray): Fan positions of shape (n_fans, 2) with integer-valued row and column indices.
        rack_loads (np.ndarray): Thermal load of each rack with shape (n_racks,).
        fan_influence_matrix (np.ndarray): Fan-to-rack influence matrix A of shape (n_racks, n_fans), column-normalised.
        zone_influence_matrix (np.ndarray): Zone-to-rack indicator matrix B of shape (n_racks, n_zones).
        thermal_coupling_matrix (np.ndarray): Rack-to-rack coupling matrix C of shape (n_racks, n_racks), symmetric with zero diagonal.
        thermal_bias (np.ndarray): Fixed per-rack temperature offset of shape (n_racks,).
        p_chiller_ref (float): Reference chiller power consumption at the reference setpoint.
        p_fan_max (float): Maximum power consumption of one fan.
        p_flow_max (float): Maximum power consumption of one zone airflow.
        gamma (float): Sensitivity of the chiller power to deviations from the reference setpoint.
        t_ref (float): Reference setpoint used by the chiller energy model.
        alpha (float): Weight of the rack's own heat term in the temperature model.
        beta (float): Weight of the neighbour coupling term in the temperature model.
        epsilon (float): Small constant that avoids division by zero in the cooling term.
        f_min (float): Minimum effective cooling guaranteed to every rack.
        t_min (float): Lower temperature limit used by the overcooling penalty.
        t_safe (float): Soft upper temperature limit used by the safety penalty.
        t_hot (float): Hard upper temperature limit used by the hotspot penalty.
        delta_t (float): Temperature scale used to normalise the safety and overcooling penalties.
        delta_t_hot (float): Temperature scale used to normalise the hotspot penalty.
        lambda_safe (float): Weight of the safety penalty in the fitness.
        lambda_hot (float): Weight of the hotspot penalty in the fitness.
        lambda_over (float): Weight of the overcooling penalty in the fitness.
        lambda_balance (float): Weight of the thermal balance penalty in the fitness.
        lambda_change (float): Weight of the change-from-baseline penalty in the fitness.
        baseline_x_phys (np.ndarray): Baseline configuration expressed in physical units, with shape (dim,).
        lower_phys (np.ndarray): Physical lower bounds aligned with baseline_x_phys, with shape (dim,).
        upper_phys (np.ndarray): Physical upper bounds aligned with baseline_x_phys, with shape (dim,).
        baseline_energy (float): Total energy consumed by the baseline configuration. Used to normalise the energy term of the fitness.
    """
    n_racks: int
    n_fans: int
    n_zones: int
    dim: int
    grid_shape: tuple[int, int]

    rack_positions: np.ndarray
    fan_positions: np.ndarray
    rack_loads: np.ndarray
    fan_influence_matrix: np.ndarray
    zone_influence_matrix: np.ndarray
    thermal_coupling_matrix: np.ndarray
    thermal_bias: np.ndarray

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

    baseline_x_phys: np.ndarray
    lower_phys: np.ndarray
    upper_phys: np.ndarray
    baseline_energy: float


# Private builders used by the default factory

def _build_rack_positions(grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Build the (row, col) integer positions of all racks in row-major order.

    Args:
        grid_shape (tuple[int, int]): Number of rows and columns of the rack grid.

    Returns:
        np.ndarray: Rack positions of shape (n_racks, 2).
    """
    rows, cols = grid_shape
    return np.array(
        [[i, j] for i in range(rows) for j in range(cols)],
        dtype=float,
    )


def _build_fan_positions(grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Place the four fans on the corners of the rack grid bounding box.

    Args:
        grid_shape (tuple[int, int]): Number of rows and columns of the rack grid.

    Returns:
        np.ndarray: Fan positions of shape (4, 2).
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
    and normalise each column so the cooling delivered by every fan sums to one.

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

    return A / A.sum(axis=0, keepdims=True)


def _build_zone_influence_matrix(rack_positions: np.ndarray, grid_shape: tuple[int, int],
                                  n_zones: int) -> np.ndarray:
    """
    Assign each rack to one of the four grid quadrants and return the one-hot matrix.

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
    Build a symmetric thermal coupling matrix C with Gaussian decay in distance
    and zero diagonal so a rack does not couple with itself in the cross term.

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
    Build a reproducible default scenario using a fixed seed for the random
    rack loads and per-rack thermal biases.

    Args:
        seed (int): Random seed used to make the scenario reproducible.

    Returns:
        DataCenterScenario: Fully built scenario with the baseline energy precomputed.
    """
    # Local import avoids a circular dependency between scenario.py and energy_model.py
    from pso.use_case.encoding import phys_to_norm
    from pso.use_case.energy_model import compute_energy

    rng = np.random.default_rng(seed)

    grid_shape = GRID_SHAPE
    n_racks = grid_shape[0] * grid_shape[1]
    n_fans = N_FANS
    n_zones = N_ZONES

    rack_positions = _build_rack_positions(grid_shape)
    fan_positions = _build_fan_positions(grid_shape)

    # Per-rack heterogeneity sampled once at construction time
    rack_loads = rng.uniform(0.7, 1.3, size=n_racks)
    thermal_bias = rng.normal(0.0, 0.3, size=n_racks)

    A = _build_fan_influence_matrix(rack_positions, fan_positions, SIGMA_FAN)
    B = _build_zone_influence_matrix(rack_positions, grid_shape, n_zones)
    C = _build_thermal_coupling_matrix(rack_positions, SIGMA_THERMAL)

    # Baseline configuration in physical units and the matching physical bounds
    baseline_x_phys = np.concatenate(
        [
            [22.0],
            [0.75] * n_fans,
            [0.70] * n_zones,
        ]
    )
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

    # Build the scenario with a placeholder baseline_energy and fill it in below
    scenario = DataCenterScenario(
        n_racks=n_racks,
        n_fans=n_fans,
        n_zones=n_zones,
        dim=1 + n_fans + n_zones,
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

    # Compute the actual baseline energy now that the scenario is otherwise ready
    x_baseline_norm = phys_to_norm(baseline_x_phys, scenario)
    scenario.baseline_energy = compute_energy(x_baseline_norm, scenario)["total_energy"]
    return scenario
