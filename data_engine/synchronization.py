"""
QuantLab Time-Series Synchronization System.

Aligns time series to UTC, handles timezone conversions, and fills missing trading sessions.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class DataSynchronizer:
    """Institutional Time-Series Synchronization Engine."""

    @staticmethod
    def align_to_utc(df: pd.DataFrame, source_timezone: str = "UTC") -> pd.DataFrame:
        """Convert DataFrame datetime index to UTC timezone."""
        if df.empty:
            return df

        df_out = df.copy()
        if isinstance(df_out.index, pd.DatetimeIndex):
            if df_out.index.tz is None:
                df_out.index = df_out.index.tz_localize(source_timezone).tz_convert("UTC")
            else:
                df_out.index = df_out.index.tz_convert("UTC")

        return df_out
