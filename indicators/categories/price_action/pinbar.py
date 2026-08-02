"""Pin Bar Price Action Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class PinBarIndicator(BaseIndicator):
    """Pin Bar (Hammer / Shooting Star) Price Action Pattern Detector."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="PinBar",
            category="PriceAction",
            description="Identifies Pin Bar rejection candlestick patterns (+1 for bullish, -1 for bearish, 0 for none).",
            dependencies=["open", "high", "low", "close"],
            parameters={"ratio_threshold": 2.0},
            outputs=["pinbar"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        ratio = self.params.get("ratio_threshold", 2.0)
        cols = {c.lower(): c for c in data.columns}
        o, h, l, c = (
            data[cols["open"]],
            data[cols["high"]],
            data[cols["low"]],
            data[cols["close"]],
        )

        body = np.abs(c - o)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l

        bullish_pin = (lower_wick >= ratio * body) & (upper_wick < body)
        bearish_pin = (upper_wick >= ratio * body) & (lower_wick < body)

        result = np.zeros(len(data), dtype=int)
        result[bullish_pin.values] = 1
        result[bearish_pin.values] = -1

        return pd.DataFrame({"pinbar": result}, index=data.index)
