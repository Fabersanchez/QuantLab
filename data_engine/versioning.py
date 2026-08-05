"""
QuantLab Dataset Version Manager & Snapshot Rollback Engine.

Provides automated dataset semantic versioning, version snapshot creation, and historical rollback.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class DatasetSnapshot:
    """Dataclass holding historical dataset snapshot payload."""

    version: str
    dataset_name: str
    dataframe: pd.DataFrame
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DataVersionManager:
    """Institutional Dataset Version Manager."""

    def __init__(self, initial_version: str = "1.0.0") -> None:
        self.current_version: str = initial_version
        self.snapshots: Dict[str, DatasetSnapshot] = {}

    def create_snapshot(self, dataset_name: str, df: pd.DataFrame) -> DatasetSnapshot:
        """Capture snapshot of active dataset under current version."""
        snap = DatasetSnapshot(version=self.current_version, dataset_name=dataset_name, dataframe=df.copy())
        self.snapshots[self.current_version] = snap
        return snap

    def bump_version(self, bump_type: str = "patch") -> str:
        """Increment dataset semantic version string."""
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

    def rollback_to(self, target_version: str) -> Optional[pd.DataFrame]:
        """Rollback active dataset state to historical snapshot version."""
        if target_version in self.snapshots:
            self.current_version = target_version
            return self.snapshots[target_version].dataframe.copy()
        return None
