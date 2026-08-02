"""
QuantLab Statistical Bootstrap Sampling Methods.

Provides random sampling with replacement, Non-overlapping Block Bootstrap,
Moving Block Bootstrap, and Stationary Bootstrap (Politis & Romano).
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Any, List, Optional
import numpy as np
import pandas as pd


class BaseBootstrapSampler(ABC):
    """Abstract Base Class for all bootstrap samplers."""

    @abstractmethod
    def sample(self, series: pd.Series, n_samples: Optional[int] = None) -> pd.Series:
        """Generate a bootstrap sample from input series.

        Args:
            series: Input pandas Series (returns, trade PnLs, or bar values).
            n_samples: Number of samples to draw (defaults to len(series)).

        Returns:
            Resampled pandas Series.
        """
        pass


class RandomReplacementBootstrap(BaseBootstrapSampler):
    """Standard Independent and Identically Distributed (IID) Bootstrap with replacement."""

    def sample(self, series: pd.Series, n_samples: Optional[int] = None) -> pd.Series:
        """Sample with uniform replacement."""
        if series.empty:
            return pd.Series(dtype=float)
        size = n_samples if n_samples is not None and n_samples > 0 else len(series)
        sampled_vals = np.random.choice(series.values, size=size, replace=True)
        return pd.Series(sampled_vals)


class BlockBootstrap(BaseBootstrapSampler):
    """Non-Overlapping Block Bootstrap (preserves short-term autocorrelation)."""

    def __init__(self, block_size: int = 10) -> None:
        """Initialize BlockBootstrap.

        Args:
            block_size: Size of contiguous blocks to sample.
        """
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = int(block_size)

    def sample(self, series: pd.Series, n_samples: Optional[int] = None) -> pd.Series:
        """Sample contiguous non-overlapping blocks with replacement."""
        if series.empty:
            return pd.Series(dtype=float)

        vals = series.values
        n = len(vals)
        target_len = n_samples if n_samples is not None and n_samples > 0 else n

        n_blocks = math.ceil(n / self.block_size)
        sampled_chunks = []

        while len(sampled_chunks) < target_len:
            block_idx = random.randint(0, max(0, n_blocks - 1))
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, n)
            chunk = vals[start_idx:end_idx]
            sampled_chunks.extend(chunk)

        result = np.array(sampled_chunks[:target_len])
        return pd.Series(result)


class MovingBlockBootstrap(BaseBootstrapSampler):
    """Moving (Overlapping) Block Bootstrap."""

    def __init__(self, block_size: int = 10) -> None:
        """Initialize MovingBlockBootstrap.

        Args:
            block_size: Size of contiguous overlapping blocks to sample.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        self.block_size = int(block_size)

    def sample(self, series: pd.Series, n_samples: Optional[int] = None) -> pd.Series:
        """Sample contiguous overlapping blocks with replacement."""
        if series.empty:
            return pd.Series(dtype=float)

        vals = series.values
        n = len(vals)
        target_len = n_samples if n_samples is not None and n_samples > 0 else n

        if n <= self.block_size:
            sampled_vals = np.random.choice(vals, size=target_len, replace=True)
            return pd.Series(sampled_vals)

        max_start = n - self.block_size
        sampled_chunks = []

        while len(sampled_chunks) < target_len:
            start_idx = random.randint(0, max_start)
            chunk = vals[start_idx : start_idx + self.block_size]
            sampled_chunks.extend(chunk)

        result = np.array(sampled_chunks[:target_len])
        return pd.Series(result)


class StationaryBootstrap(BaseBootstrapSampler):
    """Stationary Bootstrap (Politis & Romano 1994) with random geometric block lengths."""

    def __init__(self, avg_block_size: float = 10.0) -> None:
        """Initialize StationaryBootstrap.

        Args:
            avg_block_size: Expected average block length (1/p for Geometric(p)).
        """
        if avg_block_size <= 0:
            raise ValueError("avg_block_size must be positive.")
        self.p = 1.0 / float(avg_block_size)

    def sample(self, series: pd.Series, n_samples: Optional[int] = None) -> pd.Series:
        """Sample with random geometrically distributed block sizes."""
        if series.empty:
            return pd.Series(dtype=float)

        vals = series.values
        n = len(vals)
        target_len = n_samples if n_samples is not None and n_samples > 0 else n

        sampled = []
        curr_idx = random.randint(0, n - 1)

        while len(sampled) < target_len:
            sampled.append(vals[curr_idx])

            # With probability p, start a new random block; else advance to next item (wrapping around)
            if random.random() < self.p:
                curr_idx = random.randint(0, n - 1)
            else:
                curr_idx = (curr_idx + 1) % n

        return pd.Series(np.array(sampled[:target_len]))


class BootstrapSamplerFactory:
    """Factory to instantiate bootstrap samplers by identifier."""

    @staticmethod
    def create(method: str, **kwargs) -> BaseBootstrapSampler:
        """Create bootstrap sampler instance.

        Args:
            method: Identifier ('random', 'block', 'moving_block', 'stationary').
            kwargs: Constructor keyword arguments.

        Returns:
            Instance of BaseBootstrapSampler.
        """
        m = method.lower().strip()
        if m in ("random", "iid", "replacement"):
            return RandomReplacementBootstrap()
        elif m in ("block", "non_overlapping_block"):
            return BlockBootstrap(**kwargs)
        elif m in ("moving_block", "moving"):
            return MovingBlockBootstrap(**kwargs)
        elif m in ("stationary", "politis_romano"):
            return StationaryBootstrap(**kwargs)
        else:
            raise ValueError(f"Unknown bootstrap method '{method}'.")
