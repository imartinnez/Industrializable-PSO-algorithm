# @author: Íñigo Martínez Jiménez
# This module defines utility paths used to store experiment results and logs,
# providing helper functions to create output directories for each run
# and any extra subfolders needed inside them

from pathlib import Path
from datetime import datetime


# Root directory of the project
PSO_ROOT = Path(__file__).resolve().parents[1]

# Directory where experiment results will be stored
RESULTS_ROOT = PSO_ROOT / "results"

# Directory where log files will be stored
LOGS_ROOT = PSO_ROOT / "logs"


def make_run_dir(run_type: str) -> Path:
    """
    Create a new output directory for one experiment run.

    Args:
        run_type (str): Name used to identify the type of run.

    Returns:
        Path: Path to the created output directory.
    """
    # A timestamp is added so each run gets its own folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RESULTS_ROOT / f"{run_type}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def make_subdir(base: Path, name: str) -> Path:
    """
    Create a subdirectory inside a given base directory.

    Args:
        base (Path): Base directory where the subdirectory will be created.
        name (str): Name of the subdirectory.

    Returns:
        Path: Path to the created subdirectory.
    """
    path = base / name
    path.mkdir(parents=True, exist_ok=True)
    return path