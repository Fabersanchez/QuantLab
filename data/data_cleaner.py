"""
QuantLab Data Cleaner.

Provides automated cleaning, deduplication, row filtering, column normalization,
and dataset repair operations.
"""

from typing import Optional
import pandas as pd


class DataCleaner:
    """Automated market data cleaner and repair engine."""

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Convert column names to lower_snake_case."""
        cleaned = df.copy()
        cleaned.columns = [
            c.strip().lower().replace(" ", "_") for c in cleaned.columns
        ]
        return cleaned

    @staticmethod
    def drop_duplicates(
        df: pd.DataFrame, timestamp_col: str = "timestamp", keep: str = "last"
    ) -> pd.DataFrame:
        """Remove duplicate timestamp rows."""
        if timestamp_col in df.columns:
            return df.drop_duplicates(subset=[timestamp_col], keep=keep)
        return df.drop_duplicates(keep=keep)

    @staticmethod
    def sort_timestamp(
        df: pd.DataFrame, timestamp_col: str = "timestamp", ascending: bool = True
    ) -> pd.DataFrame:
        """Sort DataFrame by timestamp column."""
        if timestamp_col in df.columns:
            return df.sort_values(by=timestamp_col, ascending=ascending).reset_index(
                drop=True
            )
        return df

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        method: str = "ffill",
        fill_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """Fill missing values using specified strategy (ffill, bfill, constant)."""
        cleaned = df.copy()
        if method == "ffill":
            cleaned = cleaned.ffill()
        elif method == "bfill":
            cleaned = cleaned.bfill()
        elif method == "constant" and fill_value is not None:
            cleaned = cleaned.fillna(fill_value)
        return cleaned

    @staticmethod
    def remove_invalid_rows(
        df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """Filter out rows violating basic OHLCV physical laws or negative volumes."""
        cleaned = df.copy()
        cols = {c.lower(): c for c in cleaned.columns}

        # Ensure OHLC logic
        if all(k in cols for k in ["open", "high", "low", "close"]):
            o = cols["open"]
            h = cols["high"]
            l = cols["low"]
            c = cols["close"]

            valid_mask = (
                (cleaned[h] >= cleaned[l])
                & (cleaned[h] >= cleaned[o])
                & (cleaned[h] >= cleaned[c])
                & (cleaned[l] <= cleaned[o])
                & (cleaned[l] <= cleaned[c])
            )
            cleaned = cleaned[valid_mask]

        if "volume" in cols:
            v = cols["volume"]
            cleaned = cleaned[cleaned[v] >= 0]

        return cleaned.reset_index(drop=True)

    @classmethod
    def repair_dataset(
        cls,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        fill_missing_method: str = "ffill",
    ) -> pd.DataFrame:
        """Perform full automated repair sequence on a market dataset."""
        df_clean = cls.normalize_columns(df)
        df_clean = cls.drop_duplicates(df_clean, timestamp_col=timestamp_col)
        df_clean = cls.remove_invalid_rows(df_clean, timestamp_col=timestamp_col)
        df_clean = cls.fill_missing(df_clean, method=fill_missing_method)
        df_clean = cls.sort_timestamp(df_clean, timestamp_col=timestamp_col)
        return df_clean
