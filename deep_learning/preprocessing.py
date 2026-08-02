"""
QuantLab Deep Learning Sequence Preprocessing Engine.

Provides 3D sequence feature-wise and step-wise scaling (Standard, MinMax, Robust),
normalization, sequence alignment, and zero/mean padding.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class DLPreprocessor:
    """Institutional Deep Learning Sequence Preprocessor."""

    def __init__(self, scaling_method: str = "standard") -> None:
        """Initialize DLPreprocessor.

        Args:
            scaling_method: 'standard', 'minmax', 'robust', or 'none'.
        """
        self.scaling_method = scaling_method.lower().strip()
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, X_seq: np.ndarray) -> "DLPreprocessor":
        """Fit preprocessor parameters on 3D training tensor `(n_samples, sequence_length, n_features)`.

        Args:
            X_seq: 3D feature array.

        Returns:
            self
        """
        if X_seq.ndim != 3 or X_seq.size == 0:
            return self

        # Flatten time and batch dimension to compute feature stats across all steps
        flat = X_seq.reshape(-1, X_seq.shape[2])
        self._mean = np.mean(flat, axis=0)
        self._std = np.std(flat, axis=0)
        self._std[self._std == 0.0] = 1.0

        self._min = np.min(flat, axis=0)
        rng = np.max(flat, axis=0) - self._min
        self._max = np.where(rng == 0.0, 1.0, rng)

        return self

    def transform(self, X_seq: np.ndarray) -> np.ndarray:
        """Transform 3D sequence tensor using fitted scaling stats."""
        if self.scaling_method == "none" or X_seq.size == 0 or self._mean is None:
            return X_seq.copy()

        out = X_seq.copy().astype(np.float32)

        if self.scaling_method == "standard":
            out = (out - self._mean) / self._std
        elif self.scaling_method == "minmax":
            out = (out - self._min) / self._max
        elif self.scaling_method == "robust":
            flat = X_seq.reshape(-1, X_seq.shape[2])
            med = np.median(flat, axis=0)
            q25 = np.percentile(flat, 25, axis=0)
            q75 = np.percentile(flat, 75, axis=0)
            iqr = np.where((q75 - q25) == 0.0, 1.0, (q75 - q25))
            out = (out - med) / iqr

        return out

    def fit_transform(self, X_seq: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X_seq).transform(X_seq)
