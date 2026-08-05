"""
QuantLab Governance Version Manager & Rollback Engine.

Provides semantic versioning, version snapshot creation, historical rollback, and version diffing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VersionSnapshot:
    """Dataclass storing historical snapshot of a governance record."""

    version: str
    record_id: str
    payload: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VersionManager:
    """Institutional Version Manager for Governance Registries."""

    def __init__(self, initial_version: str = "1.0.0") -> None:
        """Initialize VersionManager.

        Args:
            initial_version: Semantic version string.
        """
        self.current_version: str = initial_version
        self.snapshots: Dict[str, VersionSnapshot] = {}

    def create_snapshot(self, record_id: str, payload: Dict[str, Any]) -> VersionSnapshot:
        """Capture snapshot of current record state under active version.

        Args:
            record_id: Record UUID.
            payload: Payload state dictionary.

        Returns:
            VersionSnapshot instance.
        """
        snap = VersionSnapshot(version=self.current_version, record_id=record_id, payload=dict(payload))
        self.snapshots[self.current_version] = snap
        return snap

    def bump_version(self, bump_type: str = "patch") -> str:
        """Increment semantic version string.

        Args:
            bump_type: One of 'major', 'minor', 'patch'.

        Returns:
            New version string.
        """
        parts = self.current_version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0

        if bump_type == "major":
            major += 1
            minor, patch = 0, 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        self.current_version = f"{major}.{minor}.{patch}"
        return self.current_version

    def rollback_to(self, target_version: str) -> Optional[Dict[str, Any]]:
        """Rollback record state to historical target version snapshot.

        Args:
            target_version: Target version string.

        Returns:
            Payload dictionary of target snapshot or None if not found.
        """
        if target_version in self.snapshots:
            self.current_version = target_version
            return dict(self.snapshots[target_version].payload)
        return None
