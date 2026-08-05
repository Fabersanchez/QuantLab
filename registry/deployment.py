"""
QuantLab Deployment Artifact Packaging Engine.

Packages approved models and strategy artifacts into production-ready deployment bundles
(ONNX, Pickle, JSON specifications) without performing automatic live deployment execution.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class DeploymentPackage:
    """Dataclass holding production deployment bundle metadata."""

    package_id: str
    model_id: str
    model_name: str
    version: str
    state: str
    artifacts_included: List[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def build_bundle(self, destination_dir: str) -> str:
        """Export deployment manifest specification bundle to target directory.

        Returns:
            Absolute file path to deployment manifest JSON.
        """
        os.makedirs(destination_dir, exist_ok=True)
        manifest_path = os.path.join(destination_dir, f"deployment_manifest_{self.version}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return os.path.abspath(manifest_path)
