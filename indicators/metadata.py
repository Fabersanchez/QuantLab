"""
QuantLab Indicator Metadata.

Defines the metadata schema for registering, documenting, and auditing
quantitative technical and market indicators.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class IndicatorMetadata:
    """Metadata specification for quantitative indicators.

    Attributes:
        name: Unique identifier for the indicator (e.g., 'RSI', 'EMA', 'VWAP').
        version: Semantic version string.
        category: Indicator classification (Trend, Momentum, Volatility, Volume,
                  PriceAction, MarketStructure, SmartMoney, Liquidity, OrderFlow,
                  Statistical, Cycle, Pattern, Custom).
        author: Creator or maintainer.
        description: Summary of indicator functionality.
        equation: Mathematical formulation or LaTeX string.
        references: Bibliographic citations or documentation links.
        dependencies: Dependent indicator names or column prerequisites.
        parameters: Configurable input parameter names and default values.
        outputs: Exported column names produced by the calculation.
        status: Operational lifecycle state ('active', 'experimental', 'deprecated').
        created_at: Creation timestamp.
    """

    name: str
    version: str = "1.0.0"
    category: str = "Custom"
    author: str = "QuantLab Engineering"
    description: str = ""
    equation: str = ""
    references: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
