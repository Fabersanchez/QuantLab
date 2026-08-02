"""
QuantLab Data Transformer.

Handles timeframe conversion (resampling), feature scaling, normalization,
standardization, rolling window generation, and train/test/validation splitting.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


class DataTransformer:
    """Financial time-series data transformer."""

    @staticmethod
    def convert_timeframe(
        df: pd.DataFrame,
        target_timeframe: str,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Resample OHLCV dataset to a target timeframe (e.g., '5min', '1h', '1D').

        Args:
            df: Source OHLCV DataFrame.
            target_timeframe: Pandas frequency string ('5min', '15min', '1h', '1d').
            timestamp_col: Name of timestamp column.

        Returns:
            Resampled DataFrame with aggregated OHLCV metrics.
        """
        df_resample = df.copy()
        if timestamp_col in df_resample.columns:
            df_resample[timestamp_col] = pd.to_datetime(df_resample[timestamp_col])
            df_resample = df_resample.set_index(timestamp_col)

        cols = {c.lower(): c for c in df_resample.columns}

        agg_dict: Dict[str, str] = {}
        if "open" in cols:
            agg_dict[cols["open"]] = "first"
        if "high" in cols:
            agg_dict[cols["high"]] = "max"
        if "low" in cols:
            agg_dict[cols["low"]] = "min"
        if "close" in cols:
            agg_dict[cols["close"]] = "last"
        if "volume" in cols:
            agg_dict[cols["volume"]] = "sum"

        resampled = df_resample.resample(target_timeframe).agg(agg_dict).dropna()
        return resampled.reset_index()

    @staticmethod
    def normalize(
        series_or_df: Union[pd.Series, pd.DataFrame],
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Union[pd.Series, pd.DataFrame]:
        """Min-Max Scaling to feature_range."""
        min_val = series_or_df.min()
        max_val = series_or_df.max()
        a, b = feature_range
        scaled = a + (series_or_df - min_val) * (b - a) / (max_val - min_val + 1e-8)
        return scaled

    @staticmethod
    def standardize(
        series_or_df: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Z-Score Standardization (mean=0, std=1)."""
        mean_val = series_or_df.mean()
        std_val = series_or_df.std()
        return (series_or_df - mean_val) / (std_val + 1e-8)

    @staticmethod
    def create_rolling_windows(
        data: Union[pd.DataFrame, np.ndarray], window_size: int, step: int = 1
    ) -> np.ndarray:
        """Generate rolling sequence windows for time-series modeling."""
        arr = np.asarray(data)
        num_windows = (len(arr) - window_size) // step + 1
        if num_windows <= 0:
            return np.array([])
        windows = [
            arr[i * step : i * step + window_size] for i in range(num_windows)
        ]
        return np.array(windows)

    @staticmethod
    def train_test_split(
        df: pd.DataFrame, train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split into train and test sets without lookahead bias."""
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    @staticmethod
    def train_val_test_split(
        df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chronological split into train, validation, and test sets."""
        train_idx = int(len(df) * train_ratio)
        val_idx = int(len(df) * (train_ratio + val_ratio))
        train_df = df.iloc[:train_idx].reset_index(drop=True)
        val_df = df.iloc[train_idx:val_idx].reset_index(drop=True)
        test_df = df.iloc[val_idx:].reset_index(drop=True)
        return train_df, val_df, test_df
