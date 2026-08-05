"""
QuantLab Enterprise Project Data Model Specification.

Defines EnterpriseProject dataclass tracking project folder paths: Research, Strategies, Experiments,
Datasets, Optimization Jobs, Models, Reports, Results, Documentation, Configurations, Metadata,
Snapshots, and Assets.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class EnterpriseProject:
    """Institutional Enterprise Project Data Model."""

    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantProject"
    path: str = "./projects/QuantProject"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    is_archived: bool = False

    # Directory layout relative paths
    research_dir: str = "Research"
    strategies_dir: str = "Strategies"
    experiments_dir: str = "Experiments"
    datasets_dir: str = "Datasets"
    optimization_dir: str = "Optimization_Jobs"
    models_dir: str = "Models"
    reports_dir: str = "Reports"
    results_dir: str = "Results"
    docs_dir: str = "Documentation"
    config_dir: str = "Configurations"
    metadata_dir: str = "Metadata"
    snapshots_dir: str = "Snapshots"
    assets_dir: str = "Assets"

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert EnterpriseProject to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnterpriseProject":
        """Reconstruct EnterpriseProject from dictionary representation."""
        return cls(**data)
