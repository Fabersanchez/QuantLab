"""
QuantLab Institutional Slippage Models.

Provides realistic price slippage models for simulated order execution.
Supports Fixed, Dynamic (volume-based), Volatility-based, Liquidity-impact, and Random slippage.
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Any, Dict, Optional


class BaseSlippageModel(ABC):
    """Abstract Base Class for all slippage models."""

    @abstractmethod
    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate execution price adjusted for slippage.

        Args:
            order_price: Intended order execution price.
            quantity: Order trade quantity.
            side: Order side ('BUY' or 'SELL').
            bar_data: Dictionary of current market bar attributes (high, low, close, volume, etc.).

        Returns:
            Adjusted execution price after slippage.
        """
        pass

    def calculate_slippage_amount(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Return absolute magnitude of price slippage."""
        exec_price = self.calculate_execution_price(order_price, quantity, side, bar_data)
        return abs(exec_price - order_price)


class FixedSlippageModel(BaseSlippageModel):
    """Fixed pips/points or fixed percentage price slippage model."""

    def __init__(self, pips: float = 0.0, percentage: float = 0.0, point_value: float = 0.0001) -> None:
        """Initialize fixed slippage model.

        Args:
            pips: Slippage in pips/points.
            percentage: Slippage as decimal fraction of price (e.g. 0.0001 = 0.01%).
            point_value: Absolute value of 1 pip (e.g. 0.0001 for EURUSD, 0.01 for USDJPY).
        """
        self._pips = max(0.0, float(pips))
        self._percentage = max(0.0, float(percentage))
        self._point_value = float(point_value)

    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Apply fixed slippage against trade direction."""
        delta = (self._pips * self._point_value) + (order_price * self._percentage)
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return order_price + delta
        else:
            return order_price - delta


class DynamicSlippageModel(BaseSlippageModel):
    """Dynamic volume-based slippage model (slippage scales with trade size relative to bar volume)."""

    def __init__(
        self, base_percentage: float = 0.0001, volume_power: float = 1.0, default_bar_volume: float = 10000.0
    ) -> None:
        """Initialize dynamic volume slippage model.

        Args:
            base_percentage: Base slippage fraction.
            volume_power: Power exponent for order size ratio.
            default_bar_volume: Fallback bar volume if volume missing in bar_data.
        """
        self._base_pct = float(base_percentage)
        self._power = float(volume_power)
        self._default_vol = float(default_bar_volume)

    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate execution price scaling with trade volume ratio."""
        bar_vol = self._default_vol
        if bar_data and "volume" in bar_data and bar_data["volume"] > 0:
            bar_vol = float(bar_data["volume"])

        vol_ratio = abs(quantity) / bar_vol
        slippage_pct = self._base_pct * (vol_ratio ** self._power)
        delta = order_price * slippage_pct

        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return order_price + delta
        else:
            return order_price - delta


class VolatilityBasedSlippageModel(BaseSlippageModel):
    """Slippage model scaled by current bar range / ATR volatility."""

    def __init__(self, volatility_factor: float = 0.1) -> None:
        """Initialize volatility-based slippage model.

        Args:
            volatility_factor: Fraction of bar high-low range applied as slippage.
        """
        self._vol_factor = float(volatility_factor)

    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Apply volatility-scaled slippage."""
        bar_range = 0.0
        if bar_data:
            high = bar_data.get("high", order_price)
            low = bar_data.get("low", order_price)
            bar_range = max(0.0, float(high - low))

        if bar_range <= 0:
            bar_range = order_price * 0.001  # Default 0.1% volatility estimate

        delta = bar_range * self._vol_factor

        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return order_price + delta
        else:
            return order_price - delta


class LiquidityBasedSlippageModel(BaseSlippageModel):
    """Square-root market impact liquidity model (Almgren-Chriss framework)."""

    def __init__(self, impact_gamma: float = 0.5, avg_daily_volume: float = 1000000.0) -> None:
        """Initialize liquidity-impact model.

        Args:
            impact_gamma: Institutional market impact constant.
            avg_daily_volume: Average daily trading volume.
        """
        self._gamma = float(impact_gamma)
        self._adv = max(1.0, float(avg_daily_volume))

    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate execution price via square-root market impact formula."""
        participation_rate = abs(quantity) / self._adv
        impact_pct = self._gamma * math.sqrt(participation_rate)
        delta = order_price * impact_pct

        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return order_price + delta
        else:
            return order_price - delta


class RandomSlippageModel(BaseSlippageModel):
    """Gaussian normal distribution random slippage model."""

    def __init__(self, mean_pips: float = 0.5, std_pips: float = 0.5, point_value: float = 0.0001) -> None:
        """Initialize random normal slippage model.

        Args:
            mean_pips: Mean slippage in pips.
            std_pips: Standard deviation of slippage in pips.
            point_value: Absolute value of 1 pip.
        """
        self._mean = float(mean_pips)
        self._std = float(std_pips)
        self._point_value = float(point_value)

    def calculate_execution_price(
        self,
        order_price: float,
        quantity: float,
        side: str,
        bar_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate execution price with random Gaussian slippage noise."""
        sampled_pips = random.gauss(self._mean, self._std)
        delta = sampled_pips * self._point_value

        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return order_price + delta
        else:
            return order_price - delta


class SlippageModelFactory:
    """Factory to instantiate slippage models from name or configuration dict."""

    @staticmethod
    def create(model_type: str, **kwargs) -> BaseSlippageModel:
        """Create slippage model instance.

        Args:
            model_type: Type identifier ('fixed', 'dynamic', 'volatility', 'liquidity', 'random').
            kwargs: Constructor keyword arguments.

        Returns:
            Instance of BaseSlippageModel.
        """
        m_type = model_type.lower().strip()
        if m_type == "fixed":
            return FixedSlippageModel(**kwargs)
        elif m_type == "dynamic":
            return DynamicSlippageModel(**kwargs)
        elif m_type in ("volatility", "volatility_based"):
            return VolatilityBasedSlippageModel(**kwargs)
        elif m_type in ("liquidity", "liquidity_based"):
            return LiquidityBasedSlippageModel(**kwargs)
        elif m_type == "random":
            return RandomSlippageModel(**kwargs)
        else:
            raise ValueError(f"Unknown slippage model type '{model_type}'.")
