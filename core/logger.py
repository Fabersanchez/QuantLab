"""
QuantLab Core Logging System.

Provides an institutional-grade logging facility with standard log levels,
custom formatting, and extensibility for future exporters (e.g., file handlers,
metrics sinks, remote telemetry).
"""

import logging
import sys
from typing import Optional


class QuantLogger:
    """Institutional logger wrapper for QuantLab components.

    Provides formatted, leveled logging capabilities across DEBUG, INFO,
    WARNING, ERROR, and CRITICAL levels.
    """

    def __init__(self, name: str = "QuantLab", level: int = logging.INFO) -> None:
        """Initialize the QuantLogger.

        Args:
            name: Logger hierarchy name.
            level: Minimum logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'DEBUG'."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'INFO'."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'WARNING'."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'ERROR'."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a message with severity 'CRITICAL'."""
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        """Set minimum logging level dynamically."""
        self._logger.setLevel(level)


def get_logger(name: str = "QuantLab") -> QuantLogger:
    """Factory function to acquire a QuantLogger instance.

    Args:
        name: Name identifier for the logger instance.

    Returns:
        Configured QuantLogger instance.
    """
    return QuantLogger(name=name)
