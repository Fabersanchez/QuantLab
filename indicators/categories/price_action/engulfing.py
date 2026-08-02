"""Engulfing Pattern Price Action Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class EngulfingPatternIndicator(BaseIndicator):
    """Bullish and Bearish Engulfing Candlestick Pattern Detector."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="EngulfingPattern",
            category="PriceAction",
            description="Detects Bullish (+1) and Bearish (-1) Engulfing candlestick patterns.",
            dependencies=["open", "close"],
            outputs=["engulfing"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in data.columns}
        o, c = data[cols["open"]], data[cols["close"]]

        prev_o, prev_c = o.shift(1), c.shift(1)

        bullish = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
        bearish = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)

        result = np.zeros(len(data), dtype=int)
        result[bullish.fillna(False).values] = 1
        result[bearish.fillna(False).values] = -1

        return pd.DataFrame({"engulfing": result}, index=data.index)
