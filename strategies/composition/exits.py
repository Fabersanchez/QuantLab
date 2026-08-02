"""Strategy Exit Rules."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseExitRule(ABC):
    """Abstract Base Class for position exit rules."""

    @abstractmethod
    def evaluate_exit(self, data: pd.DataFrame, position_series: pd.Series) -> pd.Series:
        """Evaluate exit condition and return boolean Series (True to exit)."""
        pass


class StopLossExit(BaseExitRule):
    """Fixed or ATR-based Stop-Loss Exit Rule."""

    def __init__(self, stop_pct: float = 0.02) -> None:
        self.stop_pct = stop_pct

    def evaluate_exit(self, data: pd.DataFrame, position_series: pd.Series) -> pd.Series:
        # Returns exit signal Series
        return pd.Series(False, index=data.index)


class TakeProfitExit(BaseExitRule):
    """Fixed Take-Profit Exit Rule."""

    def __init__(self, target_pct: float = 0.04) -> None:
        self.target_pct = target_pct

    def evaluate_exit(self, data: pd.DataFrame, position_series: pd.Series) -> pd.Series:
        return pd.Series(False, index=data.index)


class TrailingStopExit(BaseExitRule):
    """Dynamic Trailing Stop Exit Rule."""

    def __init__(self, trailing_pct: float = 0.015) -> None:
        self.trailing_pct = trailing_pct

    def evaluate_exit(self, data: pd.DataFrame, position_series: pd.Series) -> pd.Series:
        return pd.Series(False, index=data.index)
