"""Strategy Market Filters."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFilter(ABC):
    """Abstract Base Class for market environment filters."""

    @abstractmethod
    def filter(self, data: pd.DataFrame) -> pd.Series:
        """Filter market regime, returning boolean Series (True if trade allowed)."""
        pass


class TrendFilter(BaseFilter):
    """Filters trades based on macro trend direction."""

    def __init__(self, price_col: str = "close", trend_col: str = "sma_200", allowed_regime: str = "bullish") -> None:
        self.price_col = price_col
        self.trend_col = trend_col
        self.allowed_regime = allowed_regime.lower()

    def filter(self, data: pd.DataFrame) -> pd.Series:
        if self.price_col not in data.columns or self.trend_col not in data.columns:
            return pd.Series(True, index=data.index)

        p = data[self.price_col]
        t = data[self.trend_col]

        if self.allowed_regime == "bullish":
            return p > t
        elif self.allowed_regime == "bearish":
            return p < t
        return pd.Series(True, index=data.index)


class VolatilityFilter(BaseFilter):
    """Filters trades when volatility is outside specified bounds."""

    def __init__(self, atr_col: str = "atr_14", min_atr: float = 0.0, max_atr: float = float("inf")) -> None:
        self.atr_col = atr_col
        self.min_atr = min_atr
        self.max_atr = max_atr

    def filter(self, data: pd.DataFrame) -> pd.Series:
        if self.atr_col not in data.columns:
            return pd.Series(True, index=data.index)

        atr = data[self.atr_col]
        return (atr >= self.min_atr) & (atr <= self.max_atr)
