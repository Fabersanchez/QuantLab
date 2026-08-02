"""
QuantLab Deep Learning Sequence Builder.

Converts 2D DataFrames into 3D sequence tensors `(n_samples, sequence_length, n_features)`
for sliding window temporal sequence modeling, multi-timeframe alignment, and padding.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SequenceBuilder:
    """Sliding Window Sequence Generator for Time Series Deep Learning."""

    @staticmethod
    def create_sliding_windows(
        df: pd.DataFrame,
        sequence_length: int = 30,
        step: int = 1,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sliding 3D sequence array `(n_samples, sequence_length, n_features)`.

        Args:
            df: Input market pandas DataFrame.
            sequence_length: Number of time steps per sequence window.
            step: Sliding step stride.
            feature_cols: List of feature column names.
            target_col: Optional target column name.

        Returns:
            Tuple of (X_sequences, y_targets).
        """
        if df.empty or len(df) < sequence_length:
            return (np.empty((0, sequence_length, 0)), None)

        cols = feature_cols or [c for c in df.columns if c != target_col]
        X_data = df[cols].fillna(0.0).values
        y_data = df[target_col].values if target_col and target_col in df.columns else None

        n_samples = (len(df) - sequence_length) // step + 1

        X_seq = []
        y_seq = []

        for i in range(0, len(df) - sequence_length + 1, step):
            X_seq.append(X_data[i : i + sequence_length])
            if y_data is not None:
                # Target is the value immediately following sequence or at sequence end
                y_seq.append(y_data[i + sequence_length - 1])

        X_arr = np.array(X_seq, dtype=np.float32)
        y_arr = np.array(y_seq) if y_data is not None else None

        return (X_arr, y_arr)

    @staticmethod
    def pad_sequence(
        arr: np.ndarray, target_length: int, pad_value: float = 0.0, side: str = "pre"
    ) -> np.ndarray:
        """Pad sequence array to target_length along time dimension."""
        curr_len = arr.shape[0]
        if curr_len >= target_length:
            return arr[:target_length]

        pad_width = target_length - curr_len
        if arr.ndim == 2:
            padding = np.full((pad_width, arr.shape[1]), pad_value, dtype=arr.dtype)
        else:
            padding = np.full((pad_width,), pad_value, dtype=arr.dtype)

        if side == "pre":
            return np.vstack([padding, arr])
        else:
            return np.vstack([arr, padding])
