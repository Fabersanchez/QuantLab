"""Relative Strength Index (RSI) Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class RSIIndicator(BaseIndicator):
    """Relative Strength Index (RSI) Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="RSI",
            category="Momentum",
            description="Relative Strength Index measuring velocity and magnitude of price movements.",
            equation="RSI = 100 - \\frac{100}{1 + RS}",
            dependencies=["close"],
            parameters={"period": 14},
            outputs=["rsi_14"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        delta = data[c_col].diff()

        gain = delta.clip(lower=0)
        loss = -1.0 * delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return pd.DataFrame({f"rsi_{period}": rsi}, index=data.index)
