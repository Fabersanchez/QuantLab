"""
QuantLab Research Line Data Model Specification.

Defines ResearchLine dataclass organizing complete research projects: hypotheses,
experiments, backtests, trainings, optimizations, benchmarks, models, datasets, notes, and conclusions.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class ResearchLine:
    """Institutional Research Line Data Model."""

    line_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "VolatilityArbitrageResearch"
    hypothesis: str = "Exploiting high volatility clustering with GARCH + Reinforcement Learning."
    author: str = "QuantResearcher"
    status: str = "IN_PROGRESS"  # 'IN_PROGRESS', 'VALIDATED', 'PUBLISHED', 'ARCHIVED'
    experiments: List[str] = field(default_factory=list)
    backtests: List[str] = field(default_factory=list)
    trainings: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    benchmarks: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    conclusions: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert ResearchLine to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchLine":
        """Reconstruct ResearchLine from dictionary representation."""
        return cls(**data)
