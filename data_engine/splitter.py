"""
QuantLab Time-Series Data Splitter.

Executes Train/Test splits, In-Sample/Out-of-Sample splits, and Purged Group TimeSeries CV splits.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class DataSplitter:
    """Institutional Time-Series Data Splitter Engine."""

    @staticmethod
    def train_test_split(df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform time-sequential Train/Test split.

        Args:
            df: DataFrame to split.
            train_ratio: Ratio of training sample float (e.g. 0.80 = 80%).

        Returns:
            Tuple of (train_df, test_df).
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        n = len(df)
        split_idx = int(n * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        return train_df, test_df
