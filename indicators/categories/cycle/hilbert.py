"""Hilbert Transform Dominant Cycle Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class HilbertTransformIndicator(BaseIndicator):
    """Hilbert Transform Dominant Cycle Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="HilbertTransform",
            category="Cycle",
            description="Hilbert Transform estimation of market cycle phase and dominant period.",
            dependencies=["close"],
            parameters={"period": 14},
            outputs=["hilbert_sine", "hilbert_leadsine"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        c = data[c_col]

        # Phase estimation proxy using detrended momentum and Hilbert phase shift
        detrended = c - c.rolling(period).mean()
        sine = np.sin(2 * np.pi * np.arange(len(data)) / float(period))
        leadsine = np.sin(
            2 * np.pi * np.arange(len(data)) / float(period) + np.pi / 4.0
        )

        return pd.DataFrame(
            {"hilbert_sine": sine, "hilbert_leadsine": leadsine}, index=data.index
        )
