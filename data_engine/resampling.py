"""
QuantLab Multi-Timeframe Data Resampler.

Resamples tick or lower-tf bars to target timeframes (1s, 5s, 15s, 30s, 1m, 5m, 15m, 30m, 1h, 4h, 1D, 1W, 1M)
applying institutional OHLCV aggregation rules.
"""

from typing import Any, Dict, Optional
import pandas as pd


class DataResampler:
    """Institutional Multi-Timeframe Data Resampler."""

    TIMEFRAME_MAP: Dict[str, str] = {
        "1s": "1s",
        "5s": "5s",
        "15s": "15s",
        "30s": "30s",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1D": "1D",
        "1w": "1W",
        "1W": "1W",
        "1m_month": "1ME",
        "1M": "1ME",
    }

    @staticmethod
    def resample_ohlcv(df: pd.DataFrame, target_timeframe: str = "1h") -> pd.DataFrame:
        """Resample DataFrame to target timeframe using OHLCV aggregation rules.

        Args:
            df: Input DataFrame containing OHLCV columns and datetime index or timestamp column.
            target_timeframe: Timeframe code string (e.g. '5m', '1h', '1D').

        Returns:
            Resampled OHLCV DataFrame.
        """
        if df.empty:
            return pd.DataFrame()

        df_out = df.copy()

        # Enforce DatetimeIndex
        if not isinstance(df_out.index, pd.DatetimeIndex):
            if "timestamp" in df_out.columns:
                df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
                df_out = df_out.set_index("timestamp")
            else:
                return df

        tf_rule = DataResampler.TIMEFRAME_MAP.get(target_timeframe, target_timeframe)

        agg_dict: Dict[str, str] = {}
        cols_lower = {str(c).lower(): c for c in df_out.columns}

        if "open" in cols_lower:
            agg_dict[cols_lower["open"]] = "first"
        if "high" in cols_lower:
            agg_dict[cols_lower["high"]] = "max"
        if "low" in cols_lower:
            agg_dict[cols_lower["low"]] = "min"
        if "close" in cols_lower:
            agg_dict[cols_lower["close"]] = "last"
        if "volume" in cols_lower:
            agg_dict[cols_lower["volume"]] = "sum"

        if not agg_dict:
            resampled = df_out.resample(tf_rule).last().dropna()
        else:
            resampled = df_out.resample(tf_rule).agg(agg_dict).dropna()

        return resampled
