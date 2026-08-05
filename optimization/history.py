"""
QuantLab Optimization Execution History.

Records every iteration step, parameter combination, evaluation timing, performance metrics,
fitness score, constraint validity, running convergence curve, and top solutions leaderboard.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class IterationRecord:
    """Dataclass holding evaluation outcome for a single candidate parameter set."""

    evaluation_id: int
    iteration_index: int
    parameters: Dict[str, Any]
    fitness_score: float
    is_valid: bool
    metrics: Dict[str, Any]
    duration_sec: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    violations: List[str] = field(default_factory=list)


class OptimizationHistory:
    """Institutional Optimization History Tracking Container."""

    def __init__(self) -> None:
        """Initialize OptimizationHistory."""
        self._records: List[IterationRecord] = []
        self._lock = threading.RLock()
        self._counter: int = 0
        self._best_fitness_so_far: float = -1e9

    def add_record(
        self,
        iteration_index: int,
        parameters: Dict[str, Any],
        fitness_score: float,
        is_valid: bool,
        metrics: Dict[str, Any],
        duration_sec: float,
        violations: Optional[List[str]] = None,
    ) -> IterationRecord:
        """Record an evaluation result.

        Returns:
            Newly appended IterationRecord.
        """
        with self._lock:
            self._counter += 1
            rec = IterationRecord(
                evaluation_id=self._counter,
                iteration_index=iteration_index,
                parameters=parameters,
                fitness_score=fitness_score,
                is_valid=is_valid,
                metrics=metrics,
                duration_sec=duration_sec,
                violations=violations or [],
            )
            self._records.append(rec)
            if is_valid and fitness_score > self._best_fitness_so_far:
                self._best_fitness_so_far = fitness_score
            return rec

    def get_all_records(self) -> List[IterationRecord]:
        """Get all recorded evaluation iterations."""
        with self._lock:
            return list(self._records)

    def get_top_solutions(self, k: int = 10, valid_only: bool = True) -> List[IterationRecord]:
        """Get top k best candidate solutions sorted by fitness score.

        Args:
            k: Top k count.
            valid_only: Only include constraint-satisfying solutions.

        Returns:
            List of top IterationRecord objects.
        """
        with self._lock:
            candidates = [r for r in self._records if r.is_valid] if valid_only else self._records
            candidates_sorted = sorted(candidates, key=lambda r: r.fitness_score, reverse=True)
            return candidates_sorted[:k]

    def get_convergence_curve(self) -> List[Dict[str, Any]]:
        """Get running maximum fitness evolution across iterations."""
        with self._lock:
            curve: List[Dict[str, Any]] = []
            current_max = -1e9
            for r in self._records:
                if r.is_valid and r.fitness_score > current_max:
                    current_max = r.fitness_score
                curve.append(
                    {
                        "iteration": r.iteration_index,
                        "evaluation_id": r.evaluation_id,
                        "fitness": r.fitness_score,
                        "best_so_far": current_max if current_max > -1e8 else 0.0,
                    }
                )
            return curve

    def to_dataframe(self) -> pd.DataFrame:
        """Convert full history into flat pandas DataFrame."""
        with self._lock:
            rows = []
            for r in self._records:
                row = {
                    "evaluation_id": r.evaluation_id,
                    "iteration_index": r.iteration_index,
                    "fitness_score": r.fitness_score,
                    "is_valid": r.is_valid,
                    "duration_sec": r.duration_sec,
                    "timestamp": r.timestamp,
                }
                for p_k, p_v in r.parameters.items():
                    row[f"param_{p_k}"] = p_v
                for m_k, m_v in r.metrics.items():
                    if isinstance(m_v, (int, float, str, bool)):
                        row[f"metric_{m_k}"] = m_v
                rows.append(row)
            return pd.DataFrame(rows)
