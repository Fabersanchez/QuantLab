"""
QuantLab Registry Engine Logging System.

Provides structured, specialized logging facilities for Model Registry, Experiment Registry,
Dataset Registry, Feature Registry, Artifact Registry, versioning, rollbacks, and institutional
approval workflows.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class RegistryLogger:
    """Specialized Logger for QuantLab Governance Registry operations."""

    def __init__(self, name: str = "RegistryEngine") -> None:
        """Initialize RegistryLogger.

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

    def log_registration(self, category: str, record_id: str, name: str, version: str) -> None:
        """Log registration of a governance record."""
        self._logger.info(
            f"[REGISTRY REGISTERED] Category='{category}' | ID={record_id} | Name='{name}' | Version='{version}'"
        )

    def log_approval(self, record_id: str, old_state: str, new_state: str, approver: str) -> None:
        """Log approval state transition."""
        self._logger.info(
            f"[APPROVAL TRANSITION] ID={record_id} | State='{old_state}' -> '{new_state}' | Approver='{approver}'"
        )

    def log_rollback(self, record_id: str, target_version: str) -> None:
        """Log version rollback event."""
        self._logger.warning(f"[REGISTRY ROLLBACK] ID={record_id} | RolledBackToVersion='{target_version}'")

    def log_export(self, category: str, export_format: str, filepath: str) -> None:
        """Log exporting of registry data."""
        self._logger.info(f"[REGISTRY EXPORTED] Category='{category}' | Format={export_format} | Path='{filepath}'")

    def log_error(self, category: str, error_msg: str) -> None:
        """Log registry execution error."""
        self._logger.error(f"[REGISTRY ERROR] Category='{category}' | Error: {error_msg}")


_registry_logger_instance: Optional[RegistryLogger] = None


def get_registry_logger(name: str = "RegistryEngine") -> RegistryLogger:
    """Get singleton instance of RegistryLogger."""
    global _registry_logger_instance
    if _registry_logger_instance is None:
        _registry_logger_instance = RegistryLogger(name)
    return _registry_logger_instance
