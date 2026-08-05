"""
QuantLab Experiment Registry Engine.

Registers Backtesting, Walk Forward, Monte Carlo, Optimization, Research, Portfolio, and Benchmark
runs with execution metrics, hyperparameter configs, seeds, and system metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Dataclass holding quantitative research experiment execution metadata."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantLabExperiment"
    category: str = "Backtest"  # 'Backtest', 'WalkForward', 'MonteCarlo', 'Optimization', 'Research', 'Portfolio'
    strategy_id: str = ""
    model_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_sec: float = 0.0
    status: str = "SUCCESS"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExperimentRecord to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Reconstruct ExperimentRecord from dictionary."""
        return cls(**data)


class ExperimentRegistry:
    """Institutional Experiment Registry Engine."""

    def __init__(self) -> None:
        """Initialize ExperimentRegistry."""
        self._experiments: Dict[str, ExperimentRecord] = {}

    def register_experiment(
        self,
        name: str,
        category: str = "Backtest",
        strategy_id: str = "",
        model_id: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        duration_sec: float = 0.0,
        status: str = "SUCCESS",
        notes: str = "",
    ) -> ExperimentRecord:
        """Register experiment run instance."""
        record = ExperimentRecord(
            name=name,
            category=category,
            strategy_id=strategy_id,
            model_id=model_id,
            parameters=parameters or {},
            metrics=metrics or {},
            duration_sec=duration_sec,
            status=status,
            notes=notes,
        )
        self._experiments[record.experiment_id] = record
        return record

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Fetch ExperimentRecord by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[ExperimentRecord]:
        """List all registered experiments."""
        return list(self._experiments.values())
