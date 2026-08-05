"""
QuantLab Optimization Engine Logging System.

Provides structured, specialized logging facilities for strategy optimization,
tracking search start, iterations, candidate evaluations, fitness improvements,
new best solutions, cancellations, resumptions, and completions.
"""

from typing import Any, Dict, List, Optional
from core.logger import QuantLogger, get_logger


class OptimizationLogger:
    """Specialized Logger for QuantLab Optimization Engine operations."""

    def __init__(self, name: str = "OptimizationEngine") -> None:
        """Initialize OptimizationLogger.

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

    def log_start(self, opt_id: str, algorithm: str, total_evaluations: int) -> None:
        """Log start of an optimization process.

        Args:
            opt_id: Optimization job ID.
            algorithm: Algorithm identifier.
            total_evaluations: Target number of evaluations.
        """
        self._logger.info(
            f"[OPTIMIZATION STARTED] ID={opt_id} | Algorithm='{algorithm}' | TargetEvals={total_evaluations}"
        )

    def log_iteration(self, opt_id: str, iteration: int, current_fitness: float, best_fitness: float) -> None:
        """Log completion of an optimization iteration.

        Args:
            opt_id: Optimization job ID.
            iteration: Iteration index.
            current_fitness: Observed candidate fitness score.
            best_fitness: Best fitness achieved so far.
        """
        self._logger.debug(
            f"[OPTIMIZATION ITER] ID={opt_id} | Iter={iteration} | Fitness={current_fitness:.4f} | BestSoFar={best_fitness:.4f}"
        )

    def log_improvement(self, opt_id: str, iteration: int, old_best: float, new_best: float, params: Dict[str, Any]) -> None:
        """Log discovery of an improved candidate solution.

        Args:
            opt_id: Optimization job ID.
            iteration: Iteration index.
            old_best: Previous best fitness.
            new_best: New best fitness score.
            params: Parameters dictionary.
        """
        self._logger.info(
            f"[NEW BEST FOUND] ID={opt_id} | Iter={iteration} | Fitness: {old_best:.4f} -> {new_best:.4f} | Params={params}"
        )

    def log_pause(self, opt_id: str) -> None:
        """Log pausing of an optimization job."""
        self._logger.info(f"[OPTIMIZATION PAUSED] ID={opt_id}")

    def log_resume(self, opt_id: str) -> None:
        """Log resuming of an optimization job."""
        self._logger.info(f"[OPTIMIZATION RESUMED] ID={opt_id}")

    def log_cancellation(self, opt_id: str, reason: str = "User cancelled") -> None:
        """Log cancellation of an optimization job."""
        self._logger.warning(f"[OPTIMIZATION CANCELLED] ID={opt_id} | Reason: {reason}")

    def log_completion(self, opt_id: str, best_fitness: float, total_time_sec: float) -> None:
        """Log successful completion of optimization process."""
        self._logger.info(
            f"[OPTIMIZATION COMPLETED] ID={opt_id} | BestFitness={best_fitness:.4f} | TotalTime={total_time_sec:.2f}s"
        )

    def log_error(self, opt_id: str, error_msg: str) -> None:
        """Log execution error during optimization."""
        self._logger.error(f"[OPTIMIZATION ERROR] ID={opt_id} | Error: {error_msg}")


_optimization_logger_instance: Optional[OptimizationLogger] = None


def get_optimization_logger(name: str = "OptimizationEngine") -> OptimizationLogger:
    """Get singleton instance of OptimizationLogger."""
    global _optimization_logger_instance
    if _optimization_logger_instance is None:
        _optimization_logger_instance = OptimizationLogger(name)
    return _optimization_logger_instance
