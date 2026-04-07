# @author: Íñigo Martínez Jiménez
# This module defines a helper function to configure and return a logger,
# setting the logger level, avoiding duplicated handlers, and optionally storing
# the log messages in a file

import logging
from pathlib import Path


def setup_logging(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name (str): Name of the logger.
        log_file (Path | None): File where logs will be stored. If None, logs are only shown in the console.
        level (int): Logging level to apply.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Reuse the logger if it already exists, otherwise create a new one
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent messages from being propagated to parent loggers
    # This avoids duplicated output in some configurations
    logger.propagate = False

    # If the logger already has handlers, return it as it is
    # This avoids adding the same handlers multiple times
    if logger.handlers:
        return logger

    # Define the format used for all log messages
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Add a console handler so logs are printed on screen
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add a file handler if a log file path is provided
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger