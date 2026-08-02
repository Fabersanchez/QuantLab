"""Liquidity Pools and Equal Highs/Lows Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class LiquidityPoolsIndicator(BaseIndicator):
    """Detects Equal Highs (EQH) and Equal Lows (EQL) liquidity pools."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="LiquidityPools",
            category="Liquidity",
            description="Identifies Equal Highs (EQH: +1) and Equal Lows (EQL: -1) liquidity pools.",
            dependencies=["high", "low"],
            parameters={"tolerance": 0.001, "lookback": 20},
            outputs=["liquidity_pool"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        tol = self.params.get("tolerance", 0.001)
        lookback = self.params.get("lookback", 20)

        cols = {c.lower(): c for c in data.columns}
        h, l = data[cols["high"]], data[cols["low"]]

        eqh = np.zeros(len(data), dtype=int)
        h_vals = h.values
        l_vals = l.values

        for i in range(lookback, len(data)):
            # Check if current high matches previous peak within tolerance
            prev_max = np.max(h_vals[i - lookback : i])
            if np.abs(h_vals[i] - prev_max) / (prev_max + 1e-8) <= tol:
                eqh[i] = 1

            prev_min = np.min(l_vals[i - lookback : i])
            if np.abs(l_vals[i] - prev_min) / (prev_min + 1e-8) <= tol:
                eqh[i] = -1

        return pd.DataFrame({"liquidity_pool": eqh}, index=data.index)
