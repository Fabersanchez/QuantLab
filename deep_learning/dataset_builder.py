"""
QuantLab Financial Time Series Dataset Builder.

Constructs structured time series datasets containing 3D feature sequence tensors,
targets, timestamps, and feature metadata for deep learning model training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from deep_learning.sequence_builder import SequenceBuilder


@dataclass
class TimeSeriesDataset:
    """Dataclass encapsulating 3D feature sequence tensors and targets."""

    X_seq: np.ndarray  # Shape: (n_samples, sequence_length, n_features)
    y_target: Optional[np.ndarray] = None  # Shape: (n_samples,)
    timestamps: Optional[List[Any]] = None
    feature_names: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Return total number of sequence samples."""
        return self.X_seq.shape[0] if self.X_seq.ndim == 3 else 0

    @property
    def sequence_length(self) -> int:
        """Return sequence time steps length."""
        return self.X_seq.shape[1] if self.X_seq.ndim == 3 else 0

    @property
    def n_features(self) -> int:
        """Return number of features."""
        return self.X_seq.shape[2] if self.X_seq.ndim == 3 else 0


class DatasetBuilder:
    """Institutional Time Series Deep Learning Dataset Builder."""

    @staticmethod
    def build_dataset_from_dataframe(
        df: pd.DataFrame,
        sequence_length: int = 30,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        step: int = 1,
    ) -> TimeSeriesDataset:
        """Build TimeSeriesDataset from market DataFrame.

        Args:
            df: Input market pandas DataFrame.
            sequence_length: Number of time steps in sequence window.
            target_col: Target column name.
            feature_cols: Feature column names.
            step: Stride step size.

        Returns:
            TimeSeriesDataset instance.
        """
        cols = feature_cols or [c for c in df.columns if c != target_col]
        X_seq, y_target = SequenceBuilder.create_sliding_windows(
            df=df,
            sequence_length=sequence_length,
            step=step,
            feature_cols=cols,
            target_col=target_col,
        )

        timestamps = None
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].iloc[sequence_length - 1 : len(df) : step].tolist()

        return TimeSeriesDataset(
            X_seq=X_seq,
            y_target=y_target,
            timestamps=timestamps,
            feature_names=cols,
        )
