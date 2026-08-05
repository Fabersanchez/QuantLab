"""
QuantLab Centralized Artifact Manager Engine.

Manages trained models, charts, reports, results, matrices, snapshots, weights, configs, logs,
and exports with SHA-256 integrity verification and syncing with Registry Platform.
"""

from typing import Any, Dict, List, Optional
from registry.artifact_registry import ArtifactRecord, ArtifactRegistry
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("ArtifactManager")


class ArtifactManager:
    """Institutional Centralized Artifact Manager Engine."""

    def __init__(self, registry: Optional[ArtifactRegistry] = None) -> None:
        self.registry = registry or ArtifactRegistry()
        self._artifacts: Dict[str, ArtifactRecord] = {}

    def register_artifact(
        self,
        name: str,
        artifact_type: str = "MODEL_WEIGHTS",
        filepath: str = "",
        checksum_sha256: str = "",
        file_size_bytes: int = 0,
    ) -> ArtifactRecord:
        """Register stored artifact and sync with Registry Platform.

        Returns:
            ArtifactRecord instance.
        """
        rec = self.registry.register_artifact(
            name=name,
            artifact_type=artifact_type,
            filepath=filepath,
            checksum_sha256=checksum_sha256,
            file_size_bytes=file_size_bytes,
        )
        self._artifacts[rec.artifact_id] = rec
        logger.info(f"Registered Artifact '{name}' (ID={rec.artifact_id}, Type={artifact_type})")
        return rec

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Fetch ArtifactRecord by ID."""
        return self._artifacts.get(artifact_id) or self.registry.get_artifact(artifact_id)

    def list_artifacts(self) -> List[ArtifactRecord]:
        """List registered artifacts."""
        return self.registry.list_artifacts()
