"""
QuantLab Feature Scalers.

Provides standardized interface adapters for feature scaling, normalization,
standardization, robust scaling, and quantile transformation.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import pandas as pd


class BaseScaler(ABC):
    """Abstract Base Class for all feature scaling transformers."""

    @abstractmethod
    def fit(
        self, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> "BaseScaler":
        """Compute scaling parameters from input dataset."""
        pass

    @abstractmethod
    def transform(
        self, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Apply scaling transformation to data."""
        pass

    def fit_transform(
        self, data: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit scaler and return transformed data."""
        return self.fit(data).transform(data)


class StandardScalerAdapter(BaseScaler):
    """Standard Z-Score Scaler (mean=0, std=1)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "StandardScalerAdapter":
        arr = np.asarray(data)
        self.mean_ = np.nanmean(arr, axis=0)
        self.scale_ = np.nanstd(arr, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet.")

        if isinstance(data, pd.DataFrame):
            scaled_df = (data - self.mean_) / self.scale_
            return scaled_df
        arr = np.asarray(data)
        return (arr - self.mean_) / self.scale_


class MinMaxScalerAdapter(BaseScaler):
    """Min-Max Range Scaler (scales features to feature_range)."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "MinMaxScalerAdapter":
        arr = np.asarray(data)
        self.data_min_ = np.nanmin(arr, axis=0)
        self.data_max_ = np.nanmax(arr, axis=0)
        return self

    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler is not fitted yet.")

        a, b = self.feature_range
        diff = self.data_max_ - self.data_min_
        diff[diff == 0.0] = 1e-8

        if isinstance(data, pd.DataFrame):
            return a + (data - self.data_min_) * (b - a) / diff
        arr = np.asarray(data)
        return a + (arr - self.data_min_) * (b - a) / diff


class RobustScalerAdapter(BaseScaler):
    """Robust Scaler using median and Interquartile Range (IQR)."""

    def __init__(self) -> None:
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "RobustScalerAdapter":
        arr = np.asarray(data)
        self.center_ = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0.0] = 1.0
        self.scale_ = iqr
        return self

    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet.")

        if isinstance(data, pd.DataFrame):
            return (data - self.center_) / self.scale_
        arr = np.asarray(data)
        return (arr - self.center_) / self.scale_


class NormalizerAdapter(BaseScaler):
    """Vector Normalizer (scaling individual samples to unit norm L2)."""

    def fit(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "NormalizerAdapter":
        return self  # Stateless

    def transform(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        if isinstance(data, pd.DataFrame):
            norms = np.linalg.norm(data.values, ord=2, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            return pd.DataFrame(data.values / norms, index=data.index, columns=data.columns)
        arr = np.asarray(data)
        norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return arr / norms


class ScalerFactory:
    """Factory for acquiring feature scalers."""

    _SCALER_MAP = {
        "standard": StandardScalerAdapter,
        "minmax": MinMaxScalerAdapter,
        "robust": RobustScalerAdapter,
        "normalizer": NormalizerAdapter,
    }

    @classmethod
    def get_scaler(cls, name: str, **kwargs) -> BaseScaler:
        key = name.lower()
        if key not in cls._SCALER_MAP:
            raise ValueError(f"Unsupported scaler type: {name}")
        return cls._SCALER_MAP[key](**kwargs)
