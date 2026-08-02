"""Price Rolling Z-Score Statistical Indicator."""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class ZScoreIndicator(BaseIndicator):
    """Rolling Z-Score Statistical Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ZScore",
            category="Statistical",
            description="Rolling Price Z-Score measuring standard deviations from rolling mean.",
            dependencies=["close"],
            parameters={"period": 20},
            outputs=["zscore_20"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 20)
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        c = data[c_col]

        mean = c.rolling(window=period).mean()
        std = c.rolling(window=period).std()
        zscore = (c - mean) / (std + 1e-8)

        return pd.DataFrame({f"zscore_{period}": zscore}, index=data.index)
