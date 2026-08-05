"""
QuantLab Feature Store & Feature Registry Engine.

Registers all quantitative feature variables, data types, origin datasets, transformation pipelines,
feature importance scores, usage counts, and feature versioning.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class FeatureRecord:
    """Dataclass holding feature variable store metadata."""

    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "feature_rsi_14"
    data_type: str = "float64"
    dataset_id: str = ""
    transformation_pipeline: str = "StandardScaler(RSI(14))"
    importance_score: float = 0.0
    usage_count: int = 1
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert FeatureRecord to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureRecord":
        """Reconstruct FeatureRecord from dictionary."""
        return cls(**data)


class FeatureRegistry:
    """Institutional Feature Registry Engine."""

    def __init__(self) -> None:
        """Initialize FeatureRegistry."""
        self._features: Dict[str, FeatureRecord] = {}

    def register_feature(
        self,
        name: str,
        data_type: str = "float64",
        dataset_id: str = "",
        transformation_pipeline: str = "",
        importance_score: float = 0.0,
        version: str = "1.0.0",
    ) -> FeatureRecord:
        """Register feature variable record."""
        record = FeatureRecord(
            name=name,
            data_type=data_type,
            dataset_id=dataset_id,
            transformation_pipeline=transformation_pipeline,
            importance_score=importance_score,
            version=version,
        )
        self._features[record.feature_id] = record
        return record

    def get_feature(self, feature_id: str) -> Optional[FeatureRecord]:
        """Fetch FeatureRecord by ID."""
        return self._features.get(feature_id)

    def list_features(self) -> List[FeatureRecord]:
        """List all registered feature variables."""
        return list(self._features.values())
