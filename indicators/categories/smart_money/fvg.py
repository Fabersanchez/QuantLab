"""Fair Value Gap (FVG) Smart Money Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class FairValueGapIndicator(BaseIndicator):
    """Fair Value Gap (FVG) and Inversion FVG Detector."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="FairValueGap",
            category="SmartMoney",
            description="Identifies Bullish FVG (+1), Bearish FVG (-1), and gap sizes.",
            dependencies=["high", "low"],
            outputs=["fvg_signal", "fvg_top", "fvg_bottom"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in data.columns}
        h, l = data[cols["high"]], data[cols["low"]]

        prev2_h = h.shift(2)
        next_l = l

        prev2_l = l.shift(2)
        next_h = h

        bullish_fvg = next_l > prev2_h
        bearish_fvg = next_h < prev2_l

        signal = np.zeros(len(data), dtype=int)
        top = np.zeros(len(data))
        bottom = np.zeros(len(data))

        bull_indices = np.where(bullish_fvg.fillna(False).values)[0]
        bear_indices = np.where(bearish_fvg.fillna(False).values)[0]

        signal[bull_indices] = 1
        top[bull_indices] = next_l.values[bull_indices]
        bottom[bull_indices] = prev2_h.values[bull_indices]

        signal[bear_indices] = -1
        top[bear_indices] = prev2_l.values[bear_indices]
        bottom[bear_indices] = next_h.values[bear_indices]

        return pd.DataFrame(
            {"fvg_signal": signal, "fvg_top": top, "fvg_bottom": bottom},
            index=data.index,
        )
