"""
QuantLab Institutional Bid-Ask Spread Models.

Provides pluggable bid-ask spread simulation models during execution backtests.
Supports Fixed Spread, Variable (volatility/session expanded), Historical, and Broker Profile spreads.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseSpreadModel(ABC):
    """Abstract Base Class for all bid-ask spread models."""

    @abstractmethod
    def get_spread(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Return absolute bid-ask spread amount at timestamp.

        Args:
            timestamp: Bar/tick timestamp.
            current_price: Mid price or close price of current bar.
            bar_data: Dictionary of bar/tick properties.

        Returns:
            Absolute spread amount.
        """
        pass

    def get_bid_ask(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """Return (bid, ask) tuple based on current mid price and spread.

        Args:
            timestamp: Bar/tick timestamp.
            current_price: Mid price or close price.
            bar_data: Bar data dictionary.

        Returns:
            Tuple (bid_price, ask_price).
        """
        spread = self.get_spread(timestamp, current_price, bar_data)
        half_spread = spread / 2.0
        return (current_price - half_spread, current_price + half_spread)


class FixedSpreadModel(BaseSpreadModel):
    """Fixed bid-ask spread model in pips or absolute price units."""

    def __init__(self, pips: float = 1.0, point_value: float = 0.0001, absolute_spread: float = 0.0) -> None:
        """Initialize fixed spread model.

        Args:
            pips: Spread magnitude in pips.
            point_value: Absolute value of 1 pip (e.g. 0.0001).
            absolute_spread: Direct spread value in price units (overrides pips if > 0).
        """
        if absolute_spread > 0:
            self._spread = float(absolute_spread)
        else:
            self._spread = float(pips) * float(point_value)

    def get_spread(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Return fixed spread."""
        return self._spread


class VariableSpreadModel(BaseSpreadModel):
    """Variable spread expanding with bar volatility or market hours."""

    def __init__(
        self,
        base_pips: float = 1.0,
        volatility_multiplier: float = 0.2,
        point_value: float = 0.0001,
    ) -> None:
        """Initialize variable spread model.

        Args:
            base_pips: Minimum base spread in pips.
            volatility_multiplier: Fraction of bar high-low range added to base spread.
            point_value: Pip size.
        """
        self._base_spread = float(base_pips) * float(point_value)
        self._vol_mult = float(volatility_multiplier)
        self._point_val = float(point_value)

    def get_spread(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate dynamic variable spread."""
        extra_spread = 0.0
        if bar_data:
            high = bar_data.get("high", current_price)
            low = bar_data.get("low", current_price)
            bar_range = max(0.0, float(high - low))
            extra_spread = bar_range * self._vol_mult

        return self._base_spread + extra_spread


class HistoricalSpreadModel(BaseSpreadModel):
    """Extracts spread directly from historical dataset columns ('bid', 'ask' or 'spread')."""

    def __init__(self, fallback_spread: float = 0.0001) -> None:
        """Initialize historical spread reader.

        Args:
            fallback_spread: Default spread if historical bid/ask/spread unavailable.
        """
        self._fallback = float(fallback_spread)

    def get_spread(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Extract spread from bar_data dictionary."""
        if not bar_data:
            return self._fallback

        if "spread" in bar_data and bar_data["spread"] is not None:
            return float(bar_data["spread"])

        if "bid" in bar_data and "ask" in bar_data:
            bid = float(bar_data["bid"])
            ask = float(bar_data["ask"])
            if ask >= bid:
                return ask - bid

        return self._fallback


class BrokerSpreadModel(BaseSpreadModel):
    """Preset institutional broker typical spreads by asset symbol."""

    DEFAULT_PROFILES: Dict[str, float] = {
        "EURUSD": 0.00010,  # 1.0 pip
        "GBPUSD": 0.00015,  # 1.5 pips
        "USDJPY": 0.012,    # 1.2 pips
        "AUDUSD": 0.00012,  # 1.2 pips
        "USDCAD": 0.00015,  # 1.5 pips
        "BTCUSD": 15.0,     # $15 spread
        "ETHUSD": 1.5,      # $1.50 spread
        "XAUUSD": 0.25,     # $0.25 gold spread
        "SPX500": 0.50,     # 0.50 index points
        "DEFAULT": 0.00020, # 2.0 pips general default
    }

    def __init__(self, asset_symbol: str = "EURUSD", custom_profiles: Optional[Dict[str, float]] = None) -> None:
        """Initialize broker profile spread model.

        Args:
            asset_symbol: Instrument symbol name.
            custom_profiles: Optional custom dictionary of symbol -> spread.
        """
        self._symbol = asset_symbol.upper()
        self._profiles = self.DEFAULT_PROFILES.copy()
        if custom_profiles:
            for k, v in custom_profiles.items():
                self._profiles[k.upper()] = float(v)

    def get_spread(
        self, timestamp: Any, current_price: float, bar_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """Return preset spread for configured asset symbol."""
        return self._profiles.get(self._symbol, self._profiles["DEFAULT"])


class SpreadModelFactory:
    """Factory to instantiate spread models from name or configuration dict."""

    @staticmethod
    def create(model_type: str, **kwargs) -> BaseSpreadModel:
        """Create spread model instance.

        Args:
            model_type: Type identifier ('fixed', 'variable', 'historical', 'broker').
            kwargs: Parameters forwarded to constructor.

        Returns:
            Instance of BaseSpreadModel.
        """
        m_type = model_type.lower().strip()
        if m_type == "fixed":
            return FixedSpreadModel(**kwargs)
        elif m_type == "variable":
            return VariableSpreadModel(**kwargs)
        elif m_type == "historical":
            return HistoricalSpreadModel(**kwargs)
        elif m_type == "broker":
            return BrokerSpreadModel(**kwargs)
        else:
            raise ValueError(f"Unknown spread model type '{model_type}'.")
