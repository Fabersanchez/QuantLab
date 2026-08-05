"""
QuantLab Data Normalization & Scaling Engine.

Provides MinMax scaling, StandardScaler (Z-score), RobustScaler, Winsorization,
and Log-transformations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


class DataNormalizer:
    """Institutional Data Normalization Engine."""

    @staticmethod
    def zscore_standardize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply Z-score standardization (mean=0, std=1) on target columns."""
        df_out = df.copy()
        for col in columns:
            if col in df_out.columns:
                std = df_out[col].std()
                if std > 0:
                    df_out[col] = (df_out[col] - df_out[col].mean()) / std
        return df_out

    @staticmethod
    def minmax_scale(df: pd.DataFrame, columns: List[str], feature_range: Tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
        """Apply MinMax scaling on target columns."""
        df_out = df.copy()
        a, b = feature_range
        for col in columns:
            if col in df_out.columns:
                min_v = df_out[col].min()
                max_v = df_out[col].max()
                if max_v > min_v:
                    df_out[col] = a + (df_out[col] - min_v) * (b - a) / (max_v - min_v)
        return df_out

    @staticmethod
    def winsorize(df: pd.DataFrame, columns: List[str], limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
        """Apply Winsorization tail clipping on target columns."""
        df_out = df.copy()
        for col in columns:
            if col in df_out.columns:
                df_out[col] = stats.mstats.winsorize(df_out[col].values, limits=limits)
        return df_out
