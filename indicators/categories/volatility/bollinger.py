"""Bollinger Bands Indicator."""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="BollingerBands",
            category="Volatility",
            description="Bollinger Bands volatility envelopes around a moving average.",
            dependencies=["close"],
            parameters={"period": 20, "std_dev": 2.0},
            outputs=["bb_upper", "bb_middle", "bb_lower", "bb_bandwidth"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 20)
        std_dev = self.params.get("std_dev", 2.0)
        c_col = [c for c in data.columns if c.lower() == "close"][0]

        middle = data[c_col].rolling(window=period).mean()
        std = data[c_col].rolling(window=period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        bandwidth = (upper - lower) / (middle + 1e-8)

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_bandwidth": bandwidth,
            },
            index=data.index,
        )
