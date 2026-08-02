"""Simple Moving Average (SMA) Indicator."""

from typing import Dict, Any
import pandas as pd

from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class SMAIndicator(BaseIndicator):
    """Simple Moving Average Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SMA",
            category="Trend",
            description="Simple Moving Average of prices over a specified period.",
            equation="SMA_t = \\frac{1}{N} \\sum_{i=0}^{N-1} P_{t-i}",
            dependencies=["close"],
            parameters={"period": 14},
            outputs=["sma_14"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        sma = data[c_col].rolling(window=period).mean()
        return pd.DataFrame({f"sma_{period}": sma}, index=data.index)
