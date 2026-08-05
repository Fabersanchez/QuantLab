"""
QuantLab Data Engineering Platform Logging System.

Provides structured, specialized logging facilities for data ingestion, cleaning,
normalization, validation, resampling, feature engineering pipelines, version snapshots,
storage events, and multi-format exports.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class DataEngineLogger:
    """Specialized Logger for QuantLab Data Engineering operations."""

    def __init__(self, name: str = "DataEngine") -> None:
        """Initialize DataEngineLogger.

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

    def log_ingestion(self, dataset_name: str, source_type: str, n_rows: int) -> None:
        """Log data ingestion event."""
        self._logger.info(
            f"[DATA INGESTED] Dataset='{dataset_name}' | Source='{source_type}' | Rows={n_rows:,}"
        )

    def log_cleaning(self, dataset_name: str, nulls_filled: int, outliers_clipped: int) -> None:
        """Log data cleaning event."""
        self._logger.info(
            f"[DATA CLEANED] Dataset='{dataset_name}' | NullsFilled={nulls_filled} | OutliersClipped={outliers_clipped}"
        )

    def log_resampling(self, dataset_name: str, source_tf: str, target_tf: str, n_bars: int) -> None:
        """Log resampling event."""
        self._logger.info(
            f"[DATA RESAMPLED] Dataset='{dataset_name}' | Timeframe='{source_tf}' -> '{target_tf}' | Bars={n_bars:,}"
        )

    def log_version(self, dataset_name: str, version: str) -> None:
        """Log dataset snapshot creation."""
        self._logger.info(f"[DATA VERSION CREATED] Dataset='{dataset_name}' | Version='{version}'")

    def log_export(self, dataset_name: str, export_format: str, filepath: str) -> None:
        """Log dataset export event."""
        self._logger.info(f"[DATA EXPORTED] Dataset='{dataset_name}' | Format={export_format} | Path='{filepath}'")

    def log_error(self, dataset_name: str, error_msg: str) -> None:
        """Log data engine execution error."""
        self._logger.error(f"[DATA ERROR] Dataset='{dataset_name}' | Error: {error_msg}")


_data_engine_logger_instance: Optional[DataEngineLogger] = None


def get_data_engine_logger(name: str = "DataEngine") -> DataEngineLogger:
    """Get singleton instance of DataEngineLogger."""
    global _data_engine_logger_instance
    if _data_engine_logger_instance is None:
        _data_engine_logger_instance = DataEngineLogger(name)
    return _data_engine_logger_instance
