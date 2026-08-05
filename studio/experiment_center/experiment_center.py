"""
QuantLab Experiment Center Engine.

Provides registration, searching, reproducibility verification, and syncing with Registry Platform.
"""

from typing import Any, Dict, List, Optional
from registry.experiment_registry import ExperimentRegistry
from studio.events.event_bus import StudioEventBus
from studio.experiment_center.experiment_model import ExperimentRecord
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("ExperimentCenter")


class ExperimentCenter:
    """Institutional Experiment Center Engine."""

    def __init__(
        self,
        event_bus: Optional[StudioEventBus] = None,
        registry: Optional[ExperimentRegistry] = None,
    ) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self.registry = registry or ExperimentRegistry()
        self._experiments: Dict[str, ExperimentRecord] = {}

    def register_experiment(
        self,
        name: str,
        category: str = "Backtest",
        dataset_id: str = "",
        model_name: str = "RandomForest",
        random_seed: int = 42,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        execution_time_sec: float = 0.0,
        status: str = "SUCCESS",
    ) -> ExperimentRecord:
        """Register reproducible experiment run record."""
        record = ExperimentRecord(
            name=name,
            category=category,
            dataset_id=dataset_id,
            model_name=model_name,
            random_seed=random_seed,
            parameters=parameters or {},
            metrics=metrics or {},
            execution_time_sec=execution_time_sec,
            status=status,
        )

        self._experiments[record.experiment_id] = record

        # Sync with Registry Platform
        try:
            self.registry.register_experiment(
                name=record.name,
                category=record.category,
                parameters=record.parameters,
                metrics=record.metrics,
                duration_sec=record.execution_time_sec,
                status=record.status,
            )
        except Exception as e:
            logger.error(f"Registry experiment sync warning: {e}")

        logger.info(f"Registered Experiment '{name}' (ID={record.experiment_id}, Category={category})")
        return record

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Fetch ExperimentRecord by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[ExperimentRecord]:
        """List registered experiments."""
        return list(self._experiments.values())
