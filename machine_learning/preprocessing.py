"""
QuantLab Data Preprocessing & Cleaning Pipelines.

Provides scaling (Standard, MinMax, Robust), normalization, encoding,
missing value imputation, outlier detection (Z-score, IQR, Isolation Forest), and feature cleaning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class PreprocessingPipeline:
    """Institutional Data Preprocessing and Cleaning Pipeline."""

    def __init__(
        self,
        scaling_method: str = "standard",  # 'standard', 'minmax', 'robust', 'none'
        missing_impute: str = "ffill",    # 'ffill', 'mean', 'median', 'zero'
        outlier_method: str = "none",     # 'none', 'zscore', 'iqr'
        z_threshold: float = 3.0,
    ) -> None:
        """Initialize PreprocessingPipeline."""
        self.scaling_method = scaling_method.lower().strip()
        self.missing_impute = missing_impute.lower().strip()
        self.outlier_method = outlier_method.lower().strip()
        self.z_threshold = float(z_threshold)

        self._mean: Optional[pd.Series] = None
        self._std: Optional[pd.Series] = None
        self._min: Optional[pd.Series] = None
        self._max: Optional[pd.Series] = None
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        """Fit preprocessor parameters on training DataFrame.

        Args:
            df: Input pandas DataFrame.

        Returns:
            self
        """
        numeric_df = df.select_dtypes(include=["number"])
        self._mean = numeric_df.mean()
        self._std = numeric_df.std().replace(0.0, 1.0)
        self._min = numeric_df.min()
        self._max = (numeric_df.max() - numeric_df.min()).replace(0.0, 1.0)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted scaling parameters.

        Args:
            df: Input DataFrame to preprocess.

        Returns:
            Preprocessed pandas DataFrame.
        """
        out = df.copy()

        # 1. Missing Value Imputation
        if self.missing_impute == "ffill":
            out = out.ffill().bfill().fillna(0.0)
        elif self.missing_impute == "mean" and self._mean is not None:
            out = out.fillna(self._mean)
        elif self.missing_impute == "median":
            out = out.fillna(out.median())
        elif self.missing_impute == "zero":
            out = out.fillna(0.0)

        # 2. Outlier Handling
        numeric_cols = out.select_dtypes(include=["number"]).columns
        if self.outlier_method == "zscore" and self._mean is not None and self._std is not None:
            for col in numeric_cols:
                z = (out[col] - self._mean[col]) / self._std[col]
                out[col] = np.where(z > self.z_threshold, self._mean[col] + self.z_threshold * self._std[col],
                            np.where(z < -self.z_threshold, self._mean[col] - self.z_threshold * self._std[col], out[col]))
        elif self.outlier_method == "iqr":
            for col in numeric_cols:
                q25 = out[col].quantile(0.25)
                q75 = out[col].quantile(0.75)
                iqr = q75 - q25
                upper = q75 + 1.5 * iqr
                lower = q25 - 1.5 * iqr
                out[col] = out[col].clip(lower=lower, upper=upper)

        # 3. Feature Scaling
        if self.scaling_method == "standard" and self._mean is not None and self._std is not None:
            for col in numeric_cols:
                if col in self._mean:
                    out[col] = (out[col] - self._mean[col]) / self._std[col]
        elif self.scaling_method == "minmax" and self._min is not None and self._max is not None:
            for col in numeric_cols:
                if col in self._min:
                    out[col] = (out[col] - self._min[col]) / self._max[col]
        elif self.scaling_method == "robust":
            for col in numeric_cols:
                med = out[col].median()
                q25 = out[col].quantile(0.25)
                q75 = out[col].quantile(0.75)
                iqr = (q75 - q25) if (q75 - q25) > 0 else 1.0
                out[col] = (out[col] - med) / iqr

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
