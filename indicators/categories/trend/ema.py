"""Exponential Moving Average (EMA) Indicator."""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="EMA",
            category="Trend",
            description="Exponential Moving Average placing greater weight on recent prices.",
            equation="EMA_t = P_t \\cdot \\alpha + EMA_{t-1} \\cdot (1 - \\alpha)",
            dependencies=["close"],
            parameters={"period": 14},
            outputs=["ema_14"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        ema = data[c_col].ewm(span=period, adjust=False).mean()
        return pd.DataFrame({f"ema_{period}": ema}, index=data.index)
