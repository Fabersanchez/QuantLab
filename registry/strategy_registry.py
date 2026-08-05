"""
QuantLab Strategy Registry Engine.

Registers trading strategies, technical indicators, entry/exit rules, filters,
money management specifications, version history, and backtest results.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class StrategyRecord:
    """Dataclass holding trading strategy governance metadata."""

    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "QuantitativeStrategy"
    version: str = "1.0.0"
    author: str = "QuantLabResearcher"
    indicators: List[str] = field(default_factory=list)
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    money_management: Dict[str, Any] = field(default_factory=dict)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert StrategyRecord to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyRecord":
        """Reconstruct StrategyRecord from dictionary."""
        return cls(**data)


class StrategyRegistry:
    """Institutional Strategy Registry Engine."""

    def __init__(self) -> None:
        """Initialize StrategyRegistry."""
        self._strategies: Dict[str, StrategyRecord] = {}

    def register_strategy(
        self,
        name: str,
        version: str = "1.0.0",
        indicators: Optional[List[str]] = None,
        entry_rules: Optional[List[str]] = None,
        exit_rules: Optional[List[str]] = None,
        filters: Optional[List[str]] = None,
        money_management: Optional[Dict[str, Any]] = None,
        performance_summary: Optional[Dict[str, float]] = None,
        author: str = "QuantLabResearcher",
    ) -> StrategyRecord:
        """Register trading strategy record."""
        record = StrategyRecord(
            name=name,
            version=version,
            indicators=indicators or [],
            entry_rules=entry_rules or [],
            exit_rules=exit_rules or [],
            filters=filters or [],
            money_management=money_management or {},
            performance_summary=performance_summary or {},
            author=author,
        )
        self._strategies[record.strategy_id] = record
        return record

    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Fetch StrategyRecord by ID."""
        return self._strategies.get(strategy_id)

    def list_strategies(self) -> List[StrategyRecord]:
        """List all registered strategies."""
        return list(self._strategies.values())
