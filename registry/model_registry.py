"""
QuantLab Model Registry Engine.

Registers ML models, DL models, RL models, Ensembles, Meta models, and Experimental models with
complete governance metadata: UUID, version, author, date, framework, dataset, features,
hyperparameters, architecture, weights path, scores, approval state, SHA-256 hash, signature, dependencies,
training duration, and hardware telemetry.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

from registry.approval import ApprovalState, ApprovalWorkflow
from registry.integrity import IntegrityChecker
from registry.metadata import SystemEnvironmentMetadata
from registry.signatures import DigitalSignature
from registry.version_manager import VersionManager


@dataclass
class ModelRecord:
    """Dataclass holding complete institutional model governance metadata."""

    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantitativeModel"
    version: str = "1.0.0"
    author: str = "QuantLabResearcher"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    framework: str = "scikit-learn"  # 'scikit-learn', 'PyTorch', 'TensorFlow', 'StableBaselines'
    model_type: str = "MachineLearning"  # 'ML', 'DL', 'RL', 'Ensemble', 'Meta'
    dataset_id: str = ""
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    architecture: str = "StandardClassifier"
    weights_path: str = ""
    scores: Dict[str, float] = field(default_factory=dict)
    state: ApprovalState = ApprovalState.DRAFT
    payload_hash: str = ""
    digital_signature: str = ""
    dependencies: List[str] = field(default_factory=list)
    training_duration_sec: float = 0.0
    hardware_metadata: Dict[str, Any] = field(default_factory=lambda: SystemEnvironmentMetadata().to_dict())

    def __post_init__(self) -> None:
        """Compute payload hash and signature if missing."""
        if not self.payload_hash:
            self.payload_hash = IntegrityChecker.compute_sha256(
                f"{self.name}:{self.version}:{self.framework}:{self.hyperparameters}"
            )
        if not self.digital_signature:
            self.digital_signature = DigitalSignature.generate_signature(
                self.model_id, self.payload_hash, self.author
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelRecord to dictionary representation."""
        data = asdict(self)
        data["state"] = self.state.value if isinstance(self.state, ApprovalState) else str(self.state)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRecord":
        """Reconstruct ModelRecord from dictionary representation."""
        data_copy = dict(data)
        if "state" in data_copy and isinstance(data_copy["state"], str):
            data_copy["state"] = ApprovalState(data_copy["state"])
        return cls(**data_copy)


class ModelRegistry:
    """Institutional Model Registry Engine."""

    def __init__(self) -> None:
        """Initialize ModelRegistry."""
        self._models: Dict[str, ModelRecord] = {}
        self._workflows: Dict[str, ApprovalWorkflow] = {}
        self._version_managers: Dict[str, VersionManager] = {}

    def register_model(
        self,
        name: str,
        framework: str = "scikit-learn",
        model_type: str = "MachineLearning",
        dataset_id: str = "",
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        architecture: str = "StandardClassifier",
        weights_path: str = "",
        scores: Optional[Dict[str, float]] = None,
        author: str = "QuantLabResearcher",
    ) -> ModelRecord:
        """Register a new quantitative model instance.

        Returns:
            ModelRecord instance.
        """
        record = ModelRecord(
            name=name,
            framework=framework,
            model_type=model_type,
            dataset_id=dataset_id,
            features=features or [],
            hyperparameters=hyperparameters or {},
            architecture=architecture,
            weights_path=weights_path,
            scores=scores or {},
            author=author,
        )
        self._models[record.model_id] = record
        self._workflows[record.model_id] = ApprovalWorkflow(initial_state=ApprovalState.DRAFT)
        self._version_managers[record.model_id] = VersionManager(initial_version=record.version)
        self._version_managers[record.model_id].create_snapshot(record.model_id, record.to_dict())
        return record

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Fetch ModelRecord by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[ModelRecord]:
        """List all registered models."""
        return list(self._models.values())

    def update_approval_state(
        self, model_id: str, new_state: ApprovalState, approver: str = "SystemAdmin", comments: str = ""
    ) -> bool:
        """Update model governance approval state."""
        record = self.get_model(model_id)
        wf = self._workflows.get(model_id)
        if not record or not wf:
            return False
        success = wf.transition_to(new_state, approver=approver, comments=comments)
        if success:
            record.state = new_state
        return success
