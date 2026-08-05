"""
QuantLab Financial Asset Specification.

Defines MarketType enumeration and Asset dataclass holding detailed trading asset metadata:
symbol, market, broker, sector, asset class, timeframes, commissions, spread, swaps, volatility,
liquidity, and asset correlation vectors.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketType(str, Enum):
    """Market asset class categorization enumeration."""

    FOREX = "FOREX"
    STOCKS = "STOCKS"
    INDICES = "INDICES"
    ETF = "ETF"
    COMMODITIES = "COMMODITIES"
    CRYPTO = "CRYPTO"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    BONDS = "BONDS"
    CUSTOM = "CUSTOM"


@dataclass
class Asset:
    """Institutional Financial Asset Specification."""

    symbol: str
    name: str = ""
    market: MarketType = MarketType.FOREX
    broker: str = "GenericBroker"
    sector: str = "Financial"
    asset_class: str = "Currencies"
    available_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "1d"])
    commissions: float = 0.0
    spread: float = 0.0001
    swap_long: float = 0.0
    swap_short: float = 0.0
    volatility: float = 0.15
    liquidity: float = 1.0
    correlations: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization defaults."""
        if not self.name:
            self.name = self.symbol
        if isinstance(self.market, str):
            try:
                self.market = MarketType(self.market)
            except ValueError:
                self.market = MarketType.CUSTOM

    def to_dict(self) -> Dict[str, Any]:
        """Convert Asset to dictionary representation."""
        data = asdict(self)
        data["market"] = self.market.value if isinstance(self.market, Enum) else str(self.market)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        """Reconstruct Asset instance from dictionary representation."""
        data_copy = dict(data)
        if "market" in data_copy and isinstance(data_copy["market"], str):
            try:
                data_copy["market"] = MarketType(data_copy["market"])
            except ValueError:
                data_copy["market"] = MarketType.CUSTOM
        return cls(**data_copy)
