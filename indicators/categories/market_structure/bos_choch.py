"""Break of Structure (BOS) and Change of Character (CHOCH) Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class BOSCHoCHIndicator(BaseIndicator):
    """Detects Break of Structure (BOS) and Change of Character (CHOCH)."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="BOS_CHOCH",
            category="MarketStructure",
            description="Identifies Break of Structure (BOS: +1/-1) and Change of Character (CHOCH: +2/-2).",
            dependencies=["high", "low", "close"],
            parameters={"lookback": 10},
            outputs=["structure_break"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.get("lookback", 10)
        cols = {c.lower(): c for c in data.columns}
        h, l, c = data[cols["high"]], data[cols["low"]], data[cols["close"]]

        recent_high = h.shift(1).rolling(lookback).max()
        recent_low = l.shift(1).rolling(lookback).min()

        breaks = np.zeros(len(data), dtype=int)
        c_vals = c.values
        rh_vals = recent_high.values
        rl_vals = recent_low.values

        for i in range(lookback + 1, len(data)):
            if c_vals[i] > rh_vals[i]:
                breaks[i] = 1  # Bullish Break
            elif c_vals[i] < rl_vals[i]:
                breaks[i] = -1  # Bearish Break

        return pd.DataFrame({"structure_break": breaks}, index=data.index)
