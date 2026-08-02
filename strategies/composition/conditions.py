"""Strategy Entry Conditions."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np
import pandas as pd


class BaseCondition(ABC):
    """Abstract Base Class for strategy entry condition rules."""

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate condition rule and return boolean Series.

        Args:
            data: Input market DataFrame.

        Returns:
            pandas.Series of boolean values (True if condition met, False otherwise).
        """
        pass


class PriceThresholdCondition(BaseCondition):
    """Evaluates if price breaches a threshold column or constant value."""

    def __init__(
        self, price_col: str = "close", threshold_col_or_val: Any = "sma_20", operator: str = ">"
    ) -> None:
        self.price_col = price_col
        self.threshold = threshold_col_or_val
        self.operator = operator

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        p = data[self.price_col]
        t = data[self.threshold] if isinstance(self.threshold, str) and self.threshold in data.columns else self.threshold

        if self.operator == ">":
            return p > t
        elif self.operator == ">=":
            return p >= t
        elif self.operator == "<":
            return p < t
        elif self.operator == "<=":
            return p <= t
        elif self.operator == "==":
            return p == t
        return pd.Series(False, index=data.index)


class IndicatorCrossCondition(BaseCondition):
    """Evaluates if fast_col crosses above/below slow_col."""

    def __init__(
        self, fast_col: str = "ema_10", slow_col: str = "ema_50", direction: str = "above"
    ) -> None:
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.direction = direction

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        fast = data[self.fast_col]
        slow = data[self.slow_col]

        fast_prev = fast.shift(1)
        slow_prev = slow.shift(1)

        if self.direction == "above":
            cross = (fast_prev <= slow_prev) & (fast > slow)
        else:
            cross = (fast_prev >= slow_prev) & (fast < slow)

        return cross.fillna(False)


class CompositeCondition(BaseCondition):
    """Combines multiple conditions using AND / OR logical operators."""

    def __init__(self, conditions: List[BaseCondition], mode: str = "AND") -> None:
        self.conditions = conditions
        self.mode = mode.upper()

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series(True, index=data.index)

        result = self.conditions[0].evaluate(data)
        for cond in self.conditions[1:]:
            eval_s = cond.evaluate(data)
            if self.mode == "AND":
                result = result & eval_s
            elif self.mode == "OR":
                result = result | eval_s

        return result
