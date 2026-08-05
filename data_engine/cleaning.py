"""
QuantLab Data Cleaning Engine.

Handles missing value imputation (ffill, bfill, linear interpolation), Z-score / IQR outlier detection,
outlier clipping, and corrupt timestamp drop/fix operations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DataCleaner:
    """Institutional Data Cleaning Engine."""

    @staticmethod
    def clean_missing_values(df: pd.DataFrame, method: str = "ffill_bfill") -> Tuple[pd.DataFrame, int]:
        """Impute missing values using specified method.

        Args:
            df: DataFrame to clean.
            method: One of 'ffill_bfill', 'interpolate', 'drop'.

        Returns:
            Tuple of (cleaned_df: pd.DataFrame, nulls_imputed_count: int).
        """
        df_out = df.copy()
        nulls_count = int(df_out.isna().sum().sum())
        if nulls_count == 0:
            return df_out, 0

        if method == "interpolate":
            df_out = df_out.interpolate(method="linear").bfill().ffill()
        elif method == "drop":
            df_out = df_out.dropna()
        else:
            df_out = df_out.ffill().bfill()

        return df_out, nulls_count

    @staticmethod
    def remove_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Tuple[pd.DataFrame, int]:
        """Clip outliers beyond Z-score threshold for specified numeric columns.

        Returns:
            Tuple of (cleaned_df: pd.DataFrame, outliers_clipped_count: int).
        """
        df_out = df.copy()
        clipped_count = 0

        for col in columns:
            if col in df_out.columns:
                mean = df_out[col].mean()
                std = df_out[col].std()
                if std > 0:
                    upper = mean + threshold * std
                    lower = mean - threshold * std
                    outliers = (df_out[col] > upper) | (df_out[col] < lower)
                    clipped_count += int(outliers.sum())
                    df_out[col] = df_out[col].clip(lower=lower, upper=upper)

        return df_out, clipped_count
