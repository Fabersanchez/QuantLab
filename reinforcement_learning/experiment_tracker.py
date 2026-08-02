"""
QuantLab Reinforcement Learning - Automated RL Experiment Tracker.

Tracks episode rewards, loss curves, policy entropy, hyperparameters,
action distributions, training duration, and curriculum stage per experiment run.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd


@dataclass
class RLExperimentRun:
    """Dataclass representing a single RL experiment run record."""

    run_id: str
    experiment_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    episode_rewards: List[float]
    loss_history: List[float]
    action_distribution: Dict[int, int]
    curriculum_stages_completed: int
    metrics: Dict[str, float]
    duration_seconds: float
    author: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RLExperimentTracker:
    """Institutional RL Experiment Tracking System.

    Automatically logs all experiment metadata, episode trajectories,
    agent hyperparameters, performance metrics, and enables DataFrame export.
    """

    def __init__(self) -> None:
        """Initialize RLExperimentTracker."""
        self._runs: Dict[str, RLExperimentRun] = {}
        self._counter: int = 0

    def log_run(
        self,
        experiment_name: str,
        algorithm: str,
        hyperparameters: Dict[str, Any],
        episode_rewards: List[float],
        loss_history: Optional[List[float]] = None,
        action_distribution: Optional[Dict[int, int]] = None,
        curriculum_stages_completed: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        duration_seconds: float = 0.0,
        author: str = "QuantLabRL",
    ) -> RLExperimentRun:
        """Log a completed RL experiment run.

        Args:
            experiment_name: Experiment identifier / description.
            algorithm: RL algorithm name.
            hyperparameters: Dict of hyperparameter key-value pairs.
            episode_rewards: List of total rewards per episode.
            loss_history: List of training loss values.
            action_distribution: Dict mapping action index to occurrence count.
            curriculum_stages_completed: Number of curriculum stages completed.
            metrics: Evaluation metrics dictionary.
            duration_seconds: Total training wall-clock time.
            author: Author / team identifier.

        Returns:
            RLExperimentRun dataclass.
        """
        self._counter += 1
        run_id = f"RLRUN-{self._counter:06d}-{uuid.uuid4().hex[:6]}"

        run = RLExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            algorithm=algorithm,
            hyperparameters=hyperparameters or {},
            episode_rewards=episode_rewards or [],
            loss_history=loss_history or [],
            action_distribution=action_distribution or {},
            curriculum_stages_completed=curriculum_stages_completed,
            metrics=metrics or {},
            duration_seconds=duration_seconds,
            author=author,
        )

        self._runs[run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[RLExperimentRun]:
        """Fetch run by ID."""
        return self._runs.get(run_id)

    def list_runs(self, algorithm_filter: Optional[str] = None) -> List[RLExperimentRun]:
        """List all tracked runs with optional algorithm filter."""
        runs = list(self._runs.values())
        if algorithm_filter:
            runs = [r for r in runs if r.algorithm.upper() == algorithm_filter.upper()]
        return runs

    def to_dataframe(self) -> pd.DataFrame:
        """Export all experiment run records to pandas DataFrame."""
        if not self._runs:
            return pd.DataFrame()

        rows = []
        for run in self._runs.values():
            row = {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "algorithm": run.algorithm,
                "n_episodes": len(run.episode_rewards),
                "mean_reward": sum(run.episode_rewards) / max(len(run.episode_rewards), 1),
                "curriculum_stages": run.curriculum_stages_completed,
                "duration_seconds": run.duration_seconds,
                "author": run.author,
                "timestamp": run.timestamp,
            }
            for k, v in run.metrics.items():
                row[f"metric_{k}"] = v
            for k, v in run.hyperparameters.items():
                row[f"param_{k}"] = v
            rows.append(row)

        return pd.DataFrame(rows)

    def export_csv(self, filepath: str) -> None:
        """Export experiment tracker to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
