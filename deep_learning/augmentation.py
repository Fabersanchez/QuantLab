"""
QuantLab Financial Time Series Data Augmentation Engine.

Provides Gaussian Noise Injection, Time Warping, Magnitude Scaling,
Sequence Permutation, Window Slicing, and Mixup augmentation routines.
"""

from typing import Optional, Tuple
import numpy as np


class TimeSeriesAugmenter:
    """Institutional Data Augmentation for Financial Time Series Tensors."""

    @staticmethod
    def inject_noise(X_seq: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
        """Inject random Gaussian noise into 3D feature sequence tensor."""
        noise = np.random.normal(0, noise_std, size=X_seq.shape)
        return X_seq + noise

    @staticmethod
    def scale_magnitude(X_seq: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Scale sequence amplitude by a random uniform factor."""
        factor = np.random.uniform(scale_range[0], scale_range[1], size=(X_seq.shape[0], 1, 1))
        return X_seq * factor

    @staticmethod
    def time_warp(X_seq: np.ndarray, warp_factor: float = 0.2) -> np.ndarray:
        """Apply random time warping interpolation across sequence steps."""
        n_samples, seq_len, n_feats = X_seq.shape
        out = np.zeros_like(X_seq)

        orig_steps = np.linspace(0, 1, seq_len)
        for i in range(n_samples):
            # Generate random warped time grid
            random_offsets = np.random.uniform(-warp_factor, warp_factor, size=seq_len)
            warped_steps = np.sort(np.clip(orig_steps + random_offsets * 0.1, 0, 1))
            warped_steps[0], warped_steps[-1] = 0.0, 1.0

            for j in range(n_feats):
                out[i, :, j] = np.interp(orig_steps, warped_steps, X_seq[i, :, j])

        return out

    @staticmethod
    def permute_blocks(X_seq: np.ndarray, n_blocks: int = 3) -> np.ndarray:
        """Permute contiguous sequence time blocks."""
        n_samples, seq_len, n_feats = X_seq.shape
        out = X_seq.copy()
        block_len = seq_len // n_blocks

        if block_len < 1:
            return out

        for i in range(n_samples):
            perm = np.random.permutation(n_blocks)
            new_seq = []
            for b in perm:
                st = b * block_len
                en = st + block_len if b < n_blocks - 1 else seq_len
                new_seq.append(out[i, st:en, :])
            out[i] = np.vstack(new_seq)

        return out

    @staticmethod
    def apply_mixup(
        X_seq: np.ndarray, y: np.ndarray, alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Mixup augmentation (convex linear interpolation of sample pairs)."""
        n = X_seq.shape[0]
        if n < 2:
            return (X_seq, y)

        lam = np.random.beta(alpha, alpha)
        indices = np.random.permutation(n)

        X_mix = lam * X_seq + (1.0 - lam) * X_seq[indices]
        y_mix = lam * y + (1.0 - lam) * y[indices]

        return (X_mix, y_mix)
