"""
QuantLab Experiment Center Data Model Specification.

Defines ExperimentRecord dataclass tracking unique experiment ID, name, category, dataset_id,
model_name, random_seed, hyperparameter config, version, execution_time_sec, hardware resources,
metrics dict, execution log, artifact paths, digital signature, and reproducibility state.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentRecord:
    """Institutional Experiment Center Record."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "BacktestExperiment"
    category: str = "Backtest"  # 'Backtest', 'WalkForward', 'MonteCarlo', 'Optimization', 'ML_Training'
    dataset_id: str = ""
    model_name: str = "RandomForest"
    random_seed: int = 42
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    execution_time_sec: float = 0.0
    hardware_resources: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    digital_signature: str = ""
    status: str = "SUCCESS"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExperimentRecord to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        """Reconstruct ExperimentRecord from dictionary representation."""
        return cls(**data)
