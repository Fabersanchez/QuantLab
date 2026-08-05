"""
QuantLab Job Manager Data Model Specification.

Defines JobRecord dataclass tracking unique job UUID, job type (Backtest, WalkForward, MonteCarlo,
ML_Training, DL_Training, RL_Training, Optimization, Benchmark, Portfolio, FeatureEngineering,
Research, DatasetImport, ReportGeneration, Validation, Export), priority (1-100), status (PENDING,
RUNNING, PAUSED, CANCELLED, SUCCESS, FAILED), dependencies list, start_time, end_time, estimated_seconds,
user, workspace_id, project_id, CPU/RAM/GPU resource consumption, execution logs, errors, result payload,
and digital signature.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class JobRecord:
    """Institutional Enterprise Job Record Data Model."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantitativeJob"
    job_type: str = "Backtest"  # 'Backtest', 'WalkForward', 'MonteCarlo', 'ML_Training', 'DL_Training', 'RL_Training', 'Optimization', 'Benchmark', 'Portfolio', 'FeatureEngineering', 'Research', 'DatasetImport', 'ReportGeneration', 'Validation', 'Export'
    priority: int = 50  # 1 (lowest) to 100 (highest)
    status: str = "PENDING"  # 'PENDING', 'RUNNING', 'PAUSED', 'CANCELLED', 'SUCCESS', 'FAILED'
    dependencies: List[str] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    estimated_seconds: float = 60.0
    elapsed_seconds: float = 0.0
    progress_percent: float = 0.0
    user: str = "QuantResearcher"
    workspace_id: str = "default_workspace"
    project_id: str = ""
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    gpu_percent: float = 0.0
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    result_payload: Dict[str, Any] = field(default_factory=dict)
    digital_signature: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert JobRecord to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        """Reconstruct JobRecord from dictionary representation."""
        return cls(**data)
