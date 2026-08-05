"""
QuantLab Enterprise Workspace Data Model Specification.

Defines EnterpriseWorkspace dataclass tracking projects, datasets, strategies, models,
experiments, reports, results, configs, layouts, open panels, favorites, enabled plugins,
environment variables, navigation history, and SHA-256 checksum integrity metadata.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class EnterpriseWorkspace:
    """Institutional Enterprise Workspace Data Model."""

    workspace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantitativeWorkspace"
    path: str = "./workspace"
    version: str = "1.0.0"
    projects: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    reports: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    configurations: Dict[str, Any] = field(default_factory=dict)
    active_layout: str = "DefaultLayout"
    open_panels: List[str] = field(default_factory=lambda: ["dashboard", "explorer"])
    favorites: List[str] = field(default_factory=list)
    enabled_plugins: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    navigation_history: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksum_sha256: str = ""

    def update_checksum(self) -> str:
        """Compute and update SHA-256 integrity checksum for workspace state."""
        raw = f"{self.workspace_id}:{self.name}:{self.version}:{self.projects}:{self.datasets}:{self.strategies}".encode(
            "utf-8"
        )
        self.checksum_sha256 = hashlib.sha256(raw).hexdigest()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self.checksum_sha256

    def to_dict(self) -> Dict[str, Any]:
        """Convert EnterpriseWorkspace to dictionary representation."""
        self.update_checksum()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnterpriseWorkspace":
        """Reconstruct EnterpriseWorkspace from dictionary representation."""
        return cls(**data)
