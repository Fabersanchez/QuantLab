"""Swing Highs and Lows (HH, HL, LH, LL) Market Structure Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class SwingPointsIndicator(BaseIndicator):
    """Detects Swing Highs and Swing Lows (HH, HL, LH, LL)."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SwingPoints",
            category="MarketStructure",
            description="Identifies local Swing High (+1) and Swing Low (-1) structural points.",
            dependencies=["high", "low"],
            parameters={"left_bars": 2, "right_bars": 2},
            outputs=["swing_high", "swing_low"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        left = self.params.get("left_bars", 2)
        right = self.params.get("right_bars", 2)
        cols = {c.lower(): c for c in data.columns}
        h, l = data[cols["high"]], data[cols["low"]]

        n = len(data)
        swing_h = np.zeros(n)
        swing_l = np.zeros(n)

        h_vals = h.values
        l_vals = l.values

        for i in range(left, n - right):
            is_sh = True
            for k in range(i - left, i + right + 1):
                if k != i and h_vals[k] >= h_vals[i]:
                    is_sh = False
                    break
            if is_sh:
                swing_h[i] = h_vals[i]

            is_sl = True
            for k in range(i - left, i + right + 1):
                if k != i and l_vals[k] <= l_vals[i]:
                    is_sl = False
                    break
            if is_sl:
                swing_l[i] = l_vals[i]

        return pd.DataFrame(
            {"swing_high": swing_h, "swing_low": swing_l}, index=data.index
        )
