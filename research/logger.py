"""
QuantLab Research Engine Logging System.

Provides structured, specialized logging facilities for the Research Engine,
recording experiment lifecycles, execution benchmarks, comparisons, validations,
exports, and errors.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class ResearchLogger:
    """Specialized Logger for QuantLab Research Engine operations."""

    def __init__(self, name: str = "ResearchEngine") -> None:
        """Initialize the ResearchLogger.

        Args:
            name: Logger component identifier.
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

    def log_creation(self, exp_uuid: str, name: str, author: str) -> None:
        """Log creation of a new experiment.

        Args:
            exp_uuid: Experiment UUID.
            name: Experiment name.
            author: Experiment author.
        """
        self._logger.info(f"[EXPERIMENT CREATED] UUID={exp_uuid} | Name='{name}' | Author='{author}'")

    def log_start(self, exp_uuid: str) -> None:
        """Log execution start of an experiment.

        Args:
            exp_uuid: Experiment UUID.
        """
        self._logger.info(f"[EXPERIMENT STARTED] UUID={exp_uuid}")

    def log_pause(self, exp_uuid: str) -> None:
        """Log pausing of an experiment.

        Args:
            exp_uuid: Experiment UUID.
        """
        self._logger.info(f"[EXPERIMENT PAUSED] UUID={exp_uuid}")

    def log_resume(self, exp_uuid: str) -> None:
        """Log resuming of an experiment.

        Args:
            exp_uuid: Experiment UUID.
        """
        self._logger.info(f"[EXPERIMENT RESUMED] UUID={exp_uuid}")

    def log_completion(self, exp_uuid: str, status: str, duration_seconds: float) -> None:
        """Log completion of an experiment.

        Args:
            exp_uuid: Experiment UUID.
            status: Final status string.
            duration_seconds: Total execution time.
        """
        self._logger.info(
            f"[EXPERIMENT COMPLETED] UUID={exp_uuid} | Status='{status}' | Time={duration_seconds:.4f}s"
        )

    def log_error(self, exp_uuid: str, error_message: str) -> None:
        """Log an error during experiment execution.

        Args:
            exp_uuid: Experiment UUID.
            error_message: Error description.
        """
        self._logger.error(f"[EXPERIMENT ERROR] UUID={exp_uuid} | Error: {error_message}")

    def log_export(self, exp_uuid: str, export_format: str, filepath: str) -> None:
        """Log exporting of experiment data.

        Args:
            exp_uuid: Experiment UUID.
            export_format: Export format (e.g., CSV, JSON, PDF).
            filepath: Destination file path.
        """
        self._logger.info(f"[EXPERIMENT EXPORT] UUID={exp_uuid} | Format={export_format} | Path='{filepath}'")

    def log_comparison(self, exp_uuids: List[str], top_experiment_uuid: str) -> None:
        """Log comparison execution across experiments.

        Args:
            exp_uuids: List of experiment UUIDs compared.
            top_experiment_uuid: UUID of winning experiment.
        """
        self._logger.info(
            f"[EXPERIMENT COMPARISON] Compared {len(exp_uuids)} experiments | Top Winner UUID={top_experiment_uuid}"
        )

    def log_validation(self, exp_uuid: str, status: str, passed_count: int, failed_count: int) -> None:
        """Log validation results of an experiment.

        Args:
            exp_uuid: Experiment UUID.
            status: Validation outcome (PASSED / REJECTED).
            passed_count: Count of passed rules.
            failed_count: Count of failed rules.
        """
        self._logger.info(
            f"[EXPERIMENT VALIDATION] UUID={exp_uuid} | Outcome={status} | Passed={passed_count} | Failed={failed_count}"
        )


_research_logger_instance: Optional[ResearchLogger] = None


def get_research_logger(name: str = "ResearchEngine") -> ResearchLogger:
    """Get or create singleton instance of ResearchLogger.

    Args:
        name: Logger component name.

    Returns:
        ResearchLogger instance.
    """
    global _research_logger_instance
    if _research_logger_instance is None:
        _research_logger_instance = ResearchLogger(name)
    return _research_logger_instance
