"""SuperTrend Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class SuperTrendIndicator(BaseIndicator):
    """SuperTrend Trend-following Indicator based on ATR."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SuperTrend",
            category="Trend",
            description="SuperTrend indicator tracking trend direction and dynamic trailing stop.",
            dependencies=["high", "low", "close"],
            parameters={"period": 10, "multiplier": 3.0},
            outputs=["supertrend", "supertrend_dir"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 10)
        multiplier = self.params.get("multiplier", 3.0)

        cols = {c.lower(): c for c in data.columns}
        h, l, c = data[cols["high"]], data[cols["low"]], data[cols["close"]]

        # ATR calculation
        hl = h - l
        hc = np.abs(h - c.shift(1))
        lc = np.abs(l - c.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        hl2 = (h + l) / 2.0
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        st = np.zeros(len(data))
        st_dir = np.ones(len(data))

        c_vals = c.values
        ub_vals = upper_band.values.copy()
        lb_vals = lower_band.values.copy()

        for i in range(1, len(data)):
            if c_vals[i] > ub_vals[i - 1]:
                st_dir[i] = 1
            elif c_vals[i] < lb_vals[i - 1]:
                st_dir[i] = -1
            else:
                st_dir[i] = st_dir[i - 1]
                if st_dir[i] == 1 and lb_vals[i] < lb_vals[i - 1]:
                    lb_vals[i] = lb_vals[i - 1]
                elif st_dir[i] == -1 and ub_vals[i] > ub_vals[i - 1]:
                    ub_vals[i] = ub_vals[i - 1]

            st[i] = lb_vals[i] if st_dir[i] == 1 else ub_vals[i]

        return pd.DataFrame(
            {"supertrend": st, "supertrend_dir": st_dir}, index=data.index
        )
