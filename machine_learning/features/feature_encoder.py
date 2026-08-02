"""
QuantLab Feature Encoders.

Provides categorical feature encoding adapters: Label Encoding, One-Hot Encoding,
Ordinal Encoding, Binary Encoding, and Target Encoding.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseEncoder(ABC):
    """Abstract Base Class for feature encoding transformers."""

    @abstractmethod
    def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "BaseEncoder":
        """Compute encoding parameters from categorical data."""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding transformation."""
        pass

    def fit_transform(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit encoder and return transformed DataFrame."""
        return self.fit(data, target).transform(data)


class LabelEncoderAdapter(BaseEncoder):
    """Integer Label Encoder for categorical columns."""

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns
        self.mappings_: Dict[str, Dict[Any, int]] = {}

    def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "LabelEncoderAdapter":
        cols = (
            self.columns
            or data.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()
        )
        for col in cols:
            unique_vals = data[col].dropna().unique()
            self.mappings_[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for col, mapping in self.mappings_.items():
            if col in result.columns:
                result[col] = result[col].map(mapping).fillna(-1).astype(int)
        return result


class OneHotEncoderAdapter(BaseEncoder):
    """One-Hot Dummy Variable Encoder for categorical columns."""

    def __init__(
        self, columns: Optional[List[str]] = None, drop_first: bool = False
    ) -> None:
        self.columns = columns
        self.drop_first = drop_first
        self.categories_: Dict[str, List[Any]] = {}

    def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "OneHotEncoderAdapter":
        cols = (
            self.columns
            or data.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()
        )
        for col in cols:
            self.categories_[col] = data[col].dropna().unique().tolist()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for col in list(self.categories_.keys()):
            if col in result.columns:
                dummies = pd.get_dummies(
                    result[col], prefix=col, drop_first=self.drop_first
                )
                result = pd.concat([result.drop(columns=[col]), dummies], axis=1)
        return result


class TargetEncoderAdapter(BaseEncoder):
    """Target Mean Encoder for categorical features."""

    def __init__(
        self, columns: Optional[List[str]] = None, smoothing: float = 1.0
    ) -> None:
        self.columns = columns
        self.smoothing = smoothing
        self.target_means_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: float = 0.0

    def fit(
        self, data: pd.DataFrame, target: Optional[pd.Series] = None
    ) -> "TargetEncoderAdapter":
        if target is None:
            raise ValueError("Target series is required for TargetEncoder.")

        self.global_mean_ = float(target.mean())
        cols = (
            self.columns
            or data.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()
        )

        for col in cols:
            means = target.groupby(data[col]).mean().to_dict()
            self.target_means_[col] = means

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for col, means in self.target_means_.items():
            if col in result.columns:
                result[col] = result[col].map(means).fillna(self.global_mean_)
        return result
