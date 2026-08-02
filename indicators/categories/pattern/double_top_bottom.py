"""Double Top and Double Bottom Pattern Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class DoubleTopBottomIndicator(BaseIndicator):
    """Detects Double Top (-1) and Double Bottom (+1) Reversal Patterns."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="DoubleTopBottom",
            category="Pattern",
            description="Identifies Double Top (-1) and Double Bottom (+1) structural chart patterns.",
            dependencies=["high", "low"],
            parameters={"lookback": 15, "tolerance": 0.002},
            outputs=["double_pattern"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.get("lookback", 15)
        tol = self.params.get("tolerance", 0.002)

        cols = {c.lower(): c for c in data.columns}
        h, l = data[cols["high"]], data[cols["low"]]

        pattern = np.zeros(len(data), dtype=int)
        h_vals = h.values
        l_vals = l.values

        for i in range(lookback, len(data)):
            window_h = h_vals[i - lookback : i]
            window_l = l_vals[i - lookback : i]

            # Double Top check
            max1_idx = np.argmax(window_h)
            max1_val = window_h[max1_idx]
            if max1_idx < len(window_h) - 3:
                max2_val = np.max(window_h[max1_idx + 2 :])
                if np.abs(max1_val - max2_val) / (max1_val + 1e-8) <= tol:
                    pattern[i] = -1

            # Double Bottom check
            min1_idx = np.argmin(window_l)
            min1_val = window_l[min1_idx]
            if min1_idx < len(window_l) - 3:
                min2_val = np.min(window_l[min1_idx + 2 :])
                if np.abs(min1_val - min2_val) / (min1_val + 1e-8) <= tol:
                    pattern[i] = 1

        return pd.DataFrame({"double_pattern": pattern}, index=data.index)
