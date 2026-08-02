"""Average True Range (ATR) Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class ATRIndicator(BaseIndicator):
    """Average True Range (ATR) Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ATR",
            category="Volatility",
            description="Average True Range measuring market volatility.",
            dependencies=["high", "low", "close"],
            parameters={"period": 14},
            outputs=["atr_14"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        cols = {c.lower(): c for c in data.columns}
        h, l, c = data[cols["high"]], data[cols["low"]], data[cols["close"]]

        hl = h - l
        hc = np.abs(h - c.shift(1))
        lc = np.abs(l - c.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

        return pd.DataFrame({f"atr_{period}": atr}, index=data.index)
