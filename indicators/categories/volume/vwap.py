"""Volume Weighted Average Price (VWAP) Indicator."""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class VWAPIndicator(BaseIndicator):
    """Volume Weighted Average Price (VWAP) Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="VWAP",
            category="Volume",
            description="Volume Weighted Average Price tracking benchmark intraday price level.",
            dependencies=["high", "low", "close", "volume"],
            outputs=["vwap"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in data.columns}
        h = data[cols["high"]]
        l = data[cols["low"]]
        c = data[cols["close"]]
        v = data[cols["volume"]]

        typical_price = (h + l + c) / 3.0
        cum_tp_v = (typical_price * v).cumsum()
        cum_v = v.cumsum()

        vwap = cum_tp_v / (cum_v + 1e-8)

        return pd.DataFrame({"vwap": vwap}, index=data.index)
