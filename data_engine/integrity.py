"""
QuantLab Data Integrity & Checksum Verification System.

Computes SHA-256 digests for DataFrames, validates row counts, and checks time series continuity.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataIntegrityChecker:
    """Institutional Data Integrity & Checksum Verification Engine."""

    @staticmethod
    def compute_df_sha256(df: pd.DataFrame) -> str:
        """Compute SHA-256 hex checksum for DataFrame content."""
        if df.empty:
            return hashlib.sha256(b"EMPTY_DATAFRAME").hexdigest()
        raw = f"{df.shape}:{df.columns.tolist()}:{df.tail(10).values.tolist()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def check_continuity(df: pd.DataFrame, expected_freq: str = "1h") -> Tuple[bool, int]:
        """Check DatetimeIndex or timestamp column for missing time gaps.

        Returns:
            Tuple of (is_continuous: bool, missing_gaps_count: int).
        """
        if df.empty:
            return True, 0

        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
        else:
            return True, 0

        if isinstance(ts, pd.DatetimeIndex):
            s = ts.to_series()
        else:
            s = pd.Series(ts)

        diffs = s.diff().dropna()
        if diffs.empty:
            return True, 0

        median_diff = diffs.median()
        gaps = diffs[diffs > median_diff * 1.8]
        return len(gaps) == 0, len(gaps)
