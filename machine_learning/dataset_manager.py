"""
QuantLab Machine Learning Dataset Manager.

Manages train/validation/test dataset splits, Walk Forward sets, Monte Carlo sets,
and synthetic dataset generation (SMOTE / Gaussian perturbation).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class DatasetSplit:
    """Dataclass encapsulating feature and target splits for ML workflows."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: Optional[pd.DataFrame] = None
    y_val: Optional[pd.Series] = None
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None

    @property
    def train_size(self) -> int:
        """Return number of training samples."""
        return len(self.X_train)

    @property
    def feature_names(self) -> List[str]:
        """Return feature column names."""
        return list(self.X_train.columns)


class DatasetManager:
    """Institutional Dataset Partitioning & Lifecycle Manager."""

    @staticmethod
    def train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_pct: float = 0.2,
        val_pct: float = 0.1,
        shuffle: bool = False,
    ) -> DatasetSplit:
        """Perform train/validation/test split on features X and target y.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            test_pct: Decimal fraction for test set.
            val_pct: Decimal fraction for validation set.
            shuffle: If True, randomly shuffle samples (False for time series).

        Returns:
            DatasetSplit dataclass.
        """
        if len(X) != len(y):
            raise ValueError(f"X length ({len(X)}) must match y length ({len(y)}).")

        n = len(X)
        if shuffle:
            indices = np.random.permutation(n)
            X_data = X.iloc[indices]
            y_data = y.iloc[indices]
        else:
            X_data = X
            y_data = y

        test_size = int(n * test_pct)
        val_size = int(n * val_pct)
        train_size = n - test_size - val_size

        X_train = X_data.iloc[:train_size]
        y_train = y_data.iloc[:train_size]

        X_val = X_data.iloc[train_size : train_size + val_size] if val_size > 0 else None
        y_val = y_data.iloc[train_size : train_size + val_size] if val_size > 0 else None

        X_test = X_data.iloc[train_size + val_size :] if test_size > 0 else None
        y_test = y_data.iloc[train_size + val_size :] if test_size > 0 else None

        return DatasetSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

    @staticmethod
    def generate_synthetic_samples(
        X: pd.DataFrame, y: pd.Series, n_samples: int = 100, noise_std: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic feature and target samples via Gaussian perturbation / SMOTE style logic.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            n_samples: Number of synthetic samples to generate.
            noise_std: Standard deviation scale of feature perturbation.

        Returns:
            Tuple of (X_synthetic, y_synthetic).
        """
        if X.empty:
            return (X.copy(), y.copy())

        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X_sampled = X.iloc[indices].copy()
        y_sampled = y.iloc[indices].copy()

        # Add perturbation to numeric columns
        numeric_cols = X_sampled.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            std_col = X[col].std() if X[col].std() > 0 else 1.0
            noise = np.random.normal(0, noise_std * std_col, size=n_samples)
            X_sampled[col] += noise

        return (X_sampled, y_sampled)
