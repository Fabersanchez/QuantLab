"""Moving Average Convergence Divergence (MACD) Indicator."""

import pandas as pd
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class MACDIndicator(BaseIndicator):
    """Moving Average Convergence Divergence (MACD) Indicator."""

    @classmethod
    def metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="MACD",
            category="Momentum",
            description="MACD trend-following momentum indicator.",
            dependencies=["close"],
            parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9},
            outputs=["macd", "macd_signal", "macd_hist"],
        )

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = self.params.get("fast_period", 12)
        slow = self.params.get("slow_period", 26)
        signal = self.params.get("signal_period", 9)

        c_col = [c for c in data.columns if c.lower() == "close"][0]
        ema_fast = data[c_col].ewm(span=fast, adjust=False).mean()
        ema_slow = data[c_col].ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line

        return pd.DataFrame(
            {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
            index=data.index,
        )
