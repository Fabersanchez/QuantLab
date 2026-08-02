"""
QuantLab Strategy Metadata.

Defines the metadata record for registering, documenting, and auditing
quantitative trading strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class StrategyMetadata:
    """Metadata specification for quantitative strategies.

    Attributes:
        name: Unique identifier for the strategy.
        version: Semantic version string.
        author: Creator or maintainer.
        description: Functional summary of strategy logic.
        category: Strategy category (TrendFollowing, MeanReversion, Momentum,
                  Arbitrage, Breakout, MarketStructure, ML_Driven, Custom).
        markets: List of compatible symbols (e.g., ['EURUSD', 'BTC/USD']).
        timeframes: List of compatible bar timeframes (e.g., ['5m', '1h', '1d']).
        risk_profile: Risk limits and position sizing configuration.
        indicators_required: List of required quantitative indicators.
        features_required: List of required predictive features.
        status: Strategy operational state ('active', 'experimental', 'deprecated').
        created_at: Creation timestamp.
    """

    name: str
    version: str = "1.0.0"
    author: str = "QuantLab Engineering"
    description: str = ""
    category: str = "Custom"
    markets: List[str] = field(default_factory=lambda: ["All"])
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    risk_profile: Dict[str, Any] = field(
        default_factory=lambda: {"max_drawdown_limit": 0.15, "risk_per_trade": 0.01}
    )
    indicators_required: List[str] = field(default_factory=list)
    features_required: List[str] = field(default_factory=list)
    status: str = "active"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
