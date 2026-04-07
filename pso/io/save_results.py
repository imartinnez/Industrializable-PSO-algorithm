# @author: Íñigo Martínez Jiménez
# This module defines helper functions to save experiment results to disk,
# providing simple utilities to store data either as CSV files or JSON files.

import json
from pathlib import Path
import pandas as pd


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (Path): Output path of the CSV file.
    """
    # Create the parent directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as a JSON file.

    Args:
        data (dict): Dictionary to save.
        path (Path): Output path of the JSON file.
    """
    # Create the parent directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)