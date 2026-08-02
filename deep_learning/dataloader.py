"""
QuantLab Deep Learning DataLoader.

Manages batch iteration, memory mapping (`np.memmap`), batching, shuffling, caching,
and prefetching for 3D time series feature tensors and target arrays.
"""

import math
from typing import Generator, Optional, Tuple, Union
import numpy as np
import pandas as pd

from deep_learning.dataset_builder import TimeSeriesDataset


class DLDataLoader:
    """Institutional Time Series DataLoader."""

    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        use_memmap: bool = False,
        memmap_path: Optional[str] = None,
    ) -> None:
        """Initialize DLDataLoader.

        Args:
            dataset: TimeSeriesDataset instance.
            batch_size: Batch size integer.
            shuffle: If True, randomly shuffle samples per epoch (False for time series).
            drop_last: Drop incomplete final batch.
            use_memmap: Enable NumPy memory mapping for ultra-large datasets.
            memmap_path: File path for memmap array.
        """
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_memmap = use_memmap

        self._X = dataset.X_seq
        self._y = dataset.y_target

        if use_memmap and memmap_path:
            # Dump to memmap for memory efficiency
            fp = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=self._X.shape)
            fp[:] = self._X[:]
            self._X = fp

    def __len__(self) -> int:
        """Return total number of batches per epoch."""
        n = len(self.dataset.X_seq)
        if self.drop_last:
            return n // self.batch_size
        else:
            return math.ceil(n / self.batch_size)

    def __iter__(self) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        """Iterate over batches for one epoch."""
        n = len(self.dataset.X_seq)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        n_batches = len(self)
        for i in range(n_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            if len(batch_indices) == 0:
                continue

            X_batch = self._X[batch_indices]
            y_batch = self._y[batch_indices] if self._y is not None else None

            yield (X_batch, y_batch)
