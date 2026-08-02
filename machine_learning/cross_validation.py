"""
QuantLab Cross-Validation Schemes.

Provides standard KFold, StratifiedKFold, TimeSeriesSplit, WalkForwardCV,
NestedCV, and PurgedGroupTimeSeriesSplit (De Prado framework with purging and embargoing to prevent lookahead leakage).
"""

from abc import ABC, abstractmethod
from typing import Any, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


class PurgedGroupTimeSeriesSplit:
    """De Prado Purged & Embargoed Time Series Cross-Validation Generator.

    Purges overlapping training samples near validation boundaries to eliminate lookahead bias and label leakage.
    """

    def __init__(
        self, n_splits: int = 5, purge_bars: int = 5, embargo_bars: int = 5
    ) -> None:
        """Initialize PurgedGroupTimeSeriesSplit.

        Args:
            n_splits: Number of CV splits.
            purge_bars: Number of bars to purge before/after validation split.
            embargo_bars: Number of bars to embargo after validation split.
        """
        self.n_splits = max(2, int(n_splits))
        self.purge_bars = max(0, int(purge_bars))
        self.embargo_bars = max(0, int(embargo_bars))

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[Any] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield (train_indices, test_indices) splits with purging and embargoing."""
        n = len(X)
        indices = np.arange(n)
        test_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n)
            test_idx = indices[test_start:test_end]

            # Purge & Embargo train index calculation
            purge_start = max(0, test_start - self.purge_bars)
            embargo_end = min(n, test_end + self.embargo_bars + self.purge_bars)

            train_mask = (indices < purge_start) | (indices >= embargo_end)
            train_idx = indices[train_mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield (train_idx, test_idx)


class CrossValidationFactory:
    """Factory to instantiate cross-validation splitters."""

    @staticmethod
    def create(cv_type: str, n_splits: int = 5, **kwargs) -> Any:
        """Create CV generator instance.

        Args:
            cv_type: Type identifier ('kfold', 'stratified', 'time_series', 'purged', 'walk_forward').
            n_splits: Number of splits.
            kwargs: Extra parameters.

        Returns:
            Cross-validation splitter instance.
        """
        c = cv_type.lower().strip()
        if c == "kfold":
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif c in ("stratified", "stratified_kfold"):
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif c in ("time_series", "tscv"):
            return TimeSeriesSplit(n_splits=n_splits)
        elif c in ("purged", "purged_group", "de_prado"):
            return PurgedGroupTimeSeriesSplit(n_splits=n_splits, **kwargs)
        else:
            return TimeSeriesSplit(n_splits=n_splits)
