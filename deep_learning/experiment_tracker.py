"""
QuantLab MLOps Deep Learning Experiment Tracker.

Tracks, logs, queries, compares, and exports Deep Learning experiment runs,
neural architectures, hyperparameters, epoch loss curves, and evaluation metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class DLExperimentRun:
    """Dataclass holding details of a single Deep Learning experiment run."""

    run_id: str
    experiment_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    loss_history: Dict[str, List[float]]
    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    duration_seconds: float
    author: str = "QuantLabDL"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DLExperimentTracker:
    """Institutional MLOps Deep Learning Experiment Tracker."""

    def __init__(self) -> None:
        """Initialize DLExperimentTracker."""
        self._runs: Dict[str, DLExperimentRun] = {}
        self._counter: int = 0

    def log_run(
        self,
        experiment_name: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        loss_history: Dict[str, List[float]],
        metrics: Dict[str, float],
        dataset_info: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
        author: str = "QuantLabDL",
    ) -> DLExperimentRun:
        """Log and store a deep learning experiment run.

        Returns:
            DLExperimentRun object.
        """
        self._counter += 1
        run_id = f"DLRUN-{self._counter:06d}-{uuid.uuid4().hex[:6]}"

        run = DLExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters or {},
            loss_history=loss_history or {},
            metrics=metrics or {},
            dataset_info=dataset_info or {},
            duration_seconds=float(duration_seconds),
            author=author,
        )

        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[DLExperimentRun]:
        """Fetch experiment run by ID."""
        return self._runs.get(run_id)

    def list_runs(self, experiment_name: Optional[str] = None) -> List[DLExperimentRun]:
        """List logged experiment runs with optional experiment_name filter."""
        runs = list(self._runs.values())
        if experiment_name:
            runs = [r for r in runs if r.experiment_name.lower() == experiment_name.lower()]
        return runs

    def to_dataframe(self) -> pd.DataFrame:
        """Export experiment run history to pandas DataFrame."""
        if not self._runs:
            return pd.DataFrame()

        rows = []
        for r in self._runs.values():
            row = {
                "run_id": r.run_id,
                "experiment_name": r.experiment_name,
                "model_type": r.model_type,
                "duration_seconds": r.duration_seconds,
                "author": r.author,
                "timestamp": r.timestamp.isoformat(),
            }
            for k, v in r.metrics.items():
                row[f"metric_{k}"] = v
            for k, v in r.hyperparameters.items():
                row[f"param_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def export_csv(self, filepath: str) -> None:
        """Export experiment runs to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
