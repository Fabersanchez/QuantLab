"""
QuantLab Studio Enterprise Logging System.

Provides structured logging facilities for QuantLab Studio shell events, navigation,
services, theme changes, notifications, and telemetry.
"""

from typing import Any, Dict, Optional
from core.logger import QuantLogger, get_logger


class StudioLogger:
    """Specialized Logger for QuantLab Studio Enterprise operations."""

    def __init__(self, name: str = "QuantLabStudio") -> None:
        """Initialize StudioLogger.

        Args:
            name: Logger hierarchy name.
        """
        self._logger: QuantLogger = get_logger(name)

    @property
    def logger(self) -> QuantLogger:
        """Get underlying QuantLogger instance."""
        return self._logger

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def log_event(self, event_type: str, details: str) -> None:
        """Log studio event bus activity."""
        self._logger.info(f"[STUDIO EVENT] EventType='{event_type}' | Details={details}")

    def log_navigation(self, module_id: str, view_id: str) -> None:
        """Log navigation router activity."""
        self._logger.info(f"[STUDIO NAVIGATION] Module='{module_id}' -> View='{view_id}'")

    def log_theme_change(self, old_theme: str, new_theme: str) -> None:
        """Log theme engine change."""
        self._logger.info(f"[STUDIO THEME] Changed: '{old_theme}' -> '{new_theme}'")


_studio_logger_instance: Optional[StudioLogger] = None


def get_studio_logger(name: str = "QuantLabStudio") -> StudioLogger:
    """Get singleton instance of StudioLogger."""
    global _studio_logger_instance
    if _studio_logger_instance is None:
        _studio_logger_instance = StudioLogger(name)
    return _studio_logger_instance
