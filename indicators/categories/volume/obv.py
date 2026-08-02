"""On-Balance Volume (OBV) Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class OBVIndicator(BaseIndicator):
    """On-Balance Volume (OBV) Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="OBV",
            category="Volume",
            description="On-Balance Volume measuring cumulative buying and selling volume pressure.",
            dependencies=["close", "volume"],
            outputs=["obv"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in data.columns}
        c = data[cols["close"]]
        v = data[cols["volume"]]

        sign = np.sign(c.diff()).fillna(0)
        obv = (sign * v).cumsum()

        return pd.DataFrame({"obv": obv}, index=data.index)
