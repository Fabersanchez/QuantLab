"""
QuantLab Artifact Registry Engine.

Registers and manages stored binary and document artifacts: model weights, charts, PDF reports,
log files, configs, exported datasets, and execution payloads.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

from registry.integrity import IntegrityChecker


@dataclass
class ArtifactRecord:
    """Dataclass holding stored artifact metadata."""

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Artifact"
    artifact_type: str = "MODEL_WEIGHTS"  # 'MODEL_WEIGHTS', 'CHART', 'REPORT', 'LOG', 'CONFIG', 'EXPORT'
    filepath: str = ""
    checksum_sha256: str = ""
    file_size_bytes: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert ArtifactRecord to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        """Reconstruct ArtifactRecord from dictionary."""
        return cls(**data)


class ArtifactRegistry:
    """Institutional Artifact Registry Engine."""

    def __init__(self) -> None:
        """Initialize ArtifactRegistry."""
        self._artifacts: Dict[str, ArtifactRecord] = {}

    def register_artifact(
        self,
        name: str,
        artifact_type: str = "MODEL_WEIGHTS",
        filepath: str = "",
        checksum_sha256: str = "",
        file_size_bytes: int = 0,
    ) -> ArtifactRecord:
        """Register stored artifact record."""
        checksum = checksum_sha256
        if not checksum and filepath:
            checksum = IntegrityChecker.compute_file_sha256(filepath)

        record = ArtifactRecord(
            name=name,
            artifact_type=artifact_type,
            filepath=filepath,
            checksum_sha256=checksum,
            file_size_bytes=file_size_bytes,
        )
        self._artifacts[record.artifact_id] = record
        return record

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Fetch ArtifactRecord by ID."""
        return self._artifacts.get(artifact_id)

    def list_artifacts(self) -> List[ArtifactRecord]:
        """List all registered artifacts."""
        return list(self._artifacts.values())
