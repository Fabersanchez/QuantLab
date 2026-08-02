"""
QuantLab MLOps Model Registry.

Registers, versions, tracks, and manages status transitions ('EXPERIMENTAL', 'STAGING', 'PRODUCTION', 'ARCHIVED')
for quantitative machine learning models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelRecord:
    """Dataclass encapsulating a registered Machine Learning model record."""

    model_id: str
    name: str
    version: int
    dataset_name: str
    author: str
    model_obj: Any
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "EXPERIMENTAL"  # 'EXPERIMENTAL', 'STAGING', 'PRODUCTION', 'ARCHIVED'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelRegistry:
    """Institutional MLOps Model Registry."""

    def __init__(self) -> None:
        """Initialize ModelRegistry."""
        self._models: Dict[str, ModelRecord] = {}
        self._latest_versions: Dict[str, int] = {}

    def register_model(
        self,
        name: str,
        model: Any,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        dataset_name: str = "GenericDataset",
        author: str = "QuantLabSystem",
        status: str = "EXPERIMENTAL",
    ) -> ModelRecord:
        """Register a new trained model instance.

        Args:
            name: Model name identifier.
            model: Trained estimator object.
            hyperparameters: Hyperparameters dictionary.
            metrics: Performance metrics dictionary.
            dataset_name: Dataset identifier.
            author: Author / Engineer name.
            status: Model lifecycle status ('EXPERIMENTAL', 'STAGING', 'PRODUCTION', 'ARCHIVED').

        Returns:
            ModelRecord instance.
        """
        version = self._latest_versions.get(name, 0) + 1
        self._latest_versions[name] = version

        model_id = f"MOD-{name.upper()}-v{version}-{uuid.uuid4().hex[:6]}"

        record = ModelRecord(
            model_id=model_id,
            name=name,
            version=version,
            dataset_name=dataset_name,
            author=author,
            model_obj=model,
            hyperparameters=hyperparameters or {},
            metrics=metrics or {},
            status=status.upper(),
        )

        self._models[model_id] = record
        return record

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Fetch model record by ID."""
        return self._models.get(model_id)

    def update_status(self, model_id: str, new_status: str) -> ModelRecord:
        """Update model lifecycle status.

        Args:
            model_id: Target model ID.
            new_status: 'EXPERIMENTAL', 'STAGING', 'PRODUCTION', or 'ARCHIVED'.

        Returns:
            Updated ModelRecord.
        """
        record = self.get_model(model_id)
        if not record:
            raise KeyError(f"Model '{model_id}' not found in registry.")

        valid_statuses = ("EXPERIMENTAL", "STAGING", "PRODUCTION", "ARCHIVED")
        new_st = new_status.upper().strip()
        if new_st not in valid_statuses:
            raise ValueError(f"Invalid status '{new_status}'. Allowed: {valid_statuses}")

        record.status = new_st
        return record

    def list_models(self, status_filter: Optional[str] = None) -> List[ModelRecord]:
        """List registered models with optional status filter."""
        models = list(self._models.values())
        if status_filter:
            flt = status_filter.upper().strip()
            models = [m for m in models if m.status == flt]
        return models

    def get_best_model(self, metric_name: str = "roc_auc", higher_is_better: bool = True) -> Optional[ModelRecord]:
        """Find registered model with highest/lowest performance metric."""
        if not self._models:
            return None

        records = [m for m in self._models.values() if metric_name in m.metrics]
        if not records:
            return None

        if higher_is_better:
            return max(records, key=lambda m: m.metrics[metric_name])
        else:
            return min(records, key=lambda m: m.metrics[metric_name])
