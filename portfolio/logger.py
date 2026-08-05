"""
QuantLab Portfolio Engine Logging System.

Provides structured, specialized logging facilities for portfolio management,
capital allocations, rebalancing events, optimizations, risk alerts, simulation runs,
exports, and state changes.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class PortfolioLogger:
    """Specialized Logger for QuantLab Portfolio Engine operations."""

    def __init__(self, name: str = "PortfolioEngine") -> None:
        """Initialize PortfolioLogger.

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

    def log_creation(self, portfolio_id: str, name: str, initial_capital: float) -> None:
        """Log creation of a portfolio.

        Args:
            portfolio_id: Portfolio unique identifier.
            name: Portfolio name.
            initial_capital: Initial account capital.
        """
        self._logger.info(f"[PORTFOLIO CREATED] ID={portfolio_id} | Name='{name}' | Capital=${initial_capital:,.2f}")

    def log_rebalance(self, portfolio_id: str, trigger_type: str, new_weights: Dict[str, float]) -> None:
        """Log rebalancing execution.

        Args:
            portfolio_id: Portfolio ID.
            trigger_type: Rebalancing trigger description.
            new_weights: New target asset weights dictionary.
        """
        self._logger.info(
            f"[PORTFOLIO REBALANCED] ID={portfolio_id} | Trigger='{trigger_type}' | Weights={new_weights}"
        )

    def log_optimization(self, portfolio_id: str, objective: str, best_score: float) -> None:
        """Log completion of portfolio optimization.

        Args:
            portfolio_id: Portfolio ID.
            objective: Target optimization objective.
            best_score: Best achieved objective score.
        """
        self._logger.info(
            f"[PORTFOLIO OPTIMIZED] ID={portfolio_id} | Objective='{objective}' | BestScore={best_score:.4f}"
        )

    def log_simulation(self, portfolio_id: str, n_simulations: int, duration_sec: float) -> None:
        """Log completion of portfolio simulation runs.

        Args:
            portfolio_id: Portfolio ID.
            n_simulations: Number of simulation paths run.
            duration_sec: Execution duration seconds.
        """
        self._logger.info(
            f"[PORTFOLIO SIMULATED] ID={portfolio_id} | Paths={n_simulations} | Time={duration_sec:.2f}s"
        )

    def log_export(self, portfolio_id: str, export_format: str, filepath: str) -> None:
        """Log exporting of portfolio data."""
        self._logger.info(
            f"[PORTFOLIO EXPORTED] ID={portfolio_id} | Format={export_format} | Path='{filepath}'"
        )

    def log_error(self, portfolio_id: str, error_msg: str) -> None:
        """Log portfolio execution error."""
        self._logger.error(f"[PORTFOLIO ERROR] ID={portfolio_id} | Error: {error_msg}")


_portfolio_logger_instance: Optional[PortfolioLogger] = None


def get_portfolio_logger(name: str = "PortfolioEngine") -> PortfolioLogger:
    """Get singleton instance of PortfolioLogger."""
    global _portfolio_logger_instance
    if _portfolio_logger_instance is None:
        _portfolio_logger_instance = PortfolioLogger(name)
    return _portfolio_logger_instance
