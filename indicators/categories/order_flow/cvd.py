"""Cumulative Volume Delta (CVD) Order Flow Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class CVDIndicator(BaseIndicator):
    """Cumulative Volume Delta (CVD) Order Flow Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="CVD",
            category="OrderFlow",
            description="Cumulative Volume Delta measuring order flow aggressive buying vs selling volume.",
            dependencies=["close", "open", "volume"],
            outputs=["cvd"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in data.columns}
        c, o, v = data[cols["close"]], data[cols["open"]], data[cols["volume"]]

        delta = np.where(c >= o, v, -1.0 * v)
        cvd = pd.Series(delta, index=data.index).cumsum()

        return pd.DataFrame({"cvd": cvd}, index=data.index)
