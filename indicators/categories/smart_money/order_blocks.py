"""Institutional Order Blocks Smart Money Indicator."""

import numpy as np
import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class OrderBlocksIndicator(BaseIndicator):
    """Identifies Bullish and Bearish Order Blocks (OB)."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="OrderBlocks",
            category="SmartMoney",
            description="Identifies Institutional Bullish (+1) and Bearish (-1) Order Blocks.",
            dependencies=["open", "high", "low", "close"],
            parameters={"threshold_ratio": 1.5},
            outputs=["order_block_signal"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        ratio = self.params.get("threshold_ratio", 1.5)
        cols = {c.lower(): c for c in data.columns}
        o, h, l, c = (
            data[cols["open"]],
            data[cols["high"]],
            data[cols["low"]],
            data[cols["close"]],
        )

        body = np.abs(c - o)
        avg_body = body.rolling(10).mean()

        # Bullish OB: Last down candle prior to a strong up move
        bullish_ob = (c.shift(1) < o.shift(1)) & (body > ratio * avg_body) & (c > o)
        bearish_ob = (c.shift(1) > o.shift(1)) & (body > ratio * avg_body) & (c < o)

        signal = np.zeros(len(data), dtype=int)
        signal[bullish_ob.fillna(False).values] = 1
        signal[bearish_ob.fillna(False).values] = -1

        return pd.DataFrame({"order_block_signal": signal}, index=data.index)
