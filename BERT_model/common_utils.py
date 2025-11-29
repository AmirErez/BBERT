"""
Common utility functions used across BBERT scripts.

This module provides shared functionality to reduce code duplication.
"""

import logging
from typing import Optional
from logging import Logger


# Frame mapping used across BBERT
# Positions 0-5 correspond to frames [-1, -3, -2, +1, +3, +2]
FRAME_MAPPING = [-1, -3, -2, +1, +3, +2]


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> Logger:
    """Set up logging configuration with consistent formatting.

    Creates a logger with standardized formatting for timestamp, level, and message.
    Supports both console and file logging.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO. Default is False.
        log_file: Optional path to log file. If provided, logs to both console and file.
            If file creation fails, logs warning and continues with console-only logging.

    Returns:
        Logger: Configured logger instance with handlers for console (and optionally file).

    Example:
        >>> logger = setup_logging(verbose=True, log_file='app.log')
        >>> logger.info('Application started')
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create log file {log_file}: {e}")

    return logger


def get_predicted_frame(frame_prob_array) -> int:
    """Get predicted reading frame from probability array.

    BBERT outputs 6 frame probabilities corresponding to reading frames in order:
    [-1, -3, -2, +1, +3, +2]. This function returns the frame with highest probability.

    Args:
        frame_prob_array: Array-like of 6 probability values for the reading frames.
            Must be indexable and compatible with numpy.argmax().

    Returns:
        int: Predicted frame number (one of: -1, -3, -2, +1, +3, +2).

    Example:
        >>> import numpy as np
        >>> probs = np.array([0.1, 0.05, 0.05, 0.6, 0.15, 0.05])
        >>> get_predicted_frame(probs)
        1  # Fourth position has highest probability, maps to frame +1
    """
    import numpy as np
    return FRAME_MAPPING[np.argmax(frame_prob_array)]
