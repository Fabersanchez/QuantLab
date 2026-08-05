"""
QuantLab Master Data Preprocessor Engine.

Orchestrates cleaning, outlier removal, standardization, and scaling operations into a unified pipeline.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from data_engine.cleaning import DataCleaner
from data_engine.normalization import DataNormalizer


class DataPreprocessor:
    """Master Institutional Data Preprocessor Engine."""

    def __init__(
        self,
        impute_missing: bool = True,
        remove_outliers: bool = True,
        zscore_threshold: float = 3.0,
        normalize_method: Optional[str] = None,  # 'zscore', 'minmax', 'winsorize'
    ) -> None:
        self.impute_missing = impute_missing
        self.remove_outliers = remove_outliers
        self.zscore_threshold = zscore_threshold
        self.normalize_method = normalize_method

    def preprocess(self, df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Run unified preprocessing pipeline on DataFrame."""
        if df.empty:
            return df

        df_out = df.copy()
        if self.impute_missing:
            df_out, _ = DataCleaner.clean_missing_values(df_out, method="ffill_bfill")

        target_cols = numeric_columns or [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])]

        if self.remove_outliers and target_cols:
            df_out, _ = DataCleaner.remove_outliers_zscore(df_out, target_cols, threshold=self.zscore_threshold)

        if self.normalize_method == "zscore" and target_cols:
            df_out = DataNormalizer.zscore_standardize(df_out, target_cols)
        elif self.normalize_method == "minmax" and target_cols:
            df_out = DataNormalizer.minmax_scale(df_out, target_cols)
        elif self.normalize_method == "winsorize" and target_cols:
            df_out = DataNormalizer.winsorize(df_out, target_cols)

        return df_out
