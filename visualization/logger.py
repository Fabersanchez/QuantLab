"""
QuantLab Visualization Engine Logging System.

Provides structured, specialized logging facilities for chart rendering, updates,
theme applications, export operations, caching events, and render timing.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class VisualizationLogger:
    """Specialized Logger for QuantLab Visualization Engine operations."""

    def __init__(self, name: str = "VisualizationEngine") -> None:
        """Initialize VisualizationLogger.

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

    def log_creation(self, chart_type: str, chart_title: str) -> None:
        """Log creation of a chart figure.

        Args:
            chart_type: Identifier of chart type (e.g. 'Candlestick', 'EquityCurve').
            chart_title: Title of chart.
        """
        self._logger.info(f"[CHART CREATED] Type='{chart_type}' | Title='{chart_title}'")

    def log_render(self, chart_type: str, render_time_ms: float) -> None:
        """Log completion of chart rendering and render time.

        Args:
            chart_type: Chart type.
            render_time_ms: Render time in milliseconds.
        """
        self._logger.debug(f"[CHART RENDERED] Type='{chart_type}' | RenderTime={render_time_ms:.2f}ms")

    def log_export(self, chart_type: str, export_format: str, filepath: str) -> None:
        """Log exporting of a chart figure.

        Args:
            chart_type: Chart type.
            export_format: Export format (PNG, SVG, PDF, HTML, Markdown).
            filepath: Destination file path.
        """
        self._logger.info(f"[CHART EXPORTED] Type='{chart_type}' | Format={export_format} | Path='{filepath}'")

    def log_error(self, chart_type: str, error_msg: str) -> None:
        """Log rendering or export error."""
        self._logger.error(f"[CHART ERROR] Type='{chart_type}' | Error: {error_msg}")


_visualization_logger_instance: Optional[VisualizationLogger] = None


def get_visualization_logger(name: str = "VisualizationEngine") -> VisualizationLogger:
    """Get singleton instance of VisualizationLogger."""
    global _visualization_logger_instance
    if _visualization_logger_instance is None:
        _visualization_logger_instance = VisualizationLogger(name)
    return _visualization_logger_instance
