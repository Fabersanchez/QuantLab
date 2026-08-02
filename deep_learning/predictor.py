"""
QuantLab Deep Learning Inference Engine.

Generates sequence predictions, class probabilities, and converts neural network outputs
into quantitative trading signals (-1 Short, 0 Neutral, +1 Long).
"""

from typing import Any, Optional
import numpy as np
import pandas as pd


class DLPredictor:
    """Institutional Deep Learning Inference Engine."""

    @staticmethod
    def predict_proba(model: Any, X_seq: np.ndarray) -> np.ndarray:
        """Generate class probabilities for 3D feature sequence tensor.

        Args:
            model: Trained PyTorch nn.Module or fallback model.
            X_seq: 3D feature array `(n_samples, sequence_length, n_features)`.

        Returns:
            1D numpy array of probabilities.
        """
        if X_seq.size == 0:
            return np.array([])

        try:
            import torch
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    X_t = torch.tensor(X_seq, dtype=torch.float32)
                    out = model(X_t)
                    return torch.sigmoid(out).cpu().numpy().ravel()
        except Exception:
            pass

        res = model(X_seq)
        return np.ravel(res)

    @staticmethod
    def predict(model: Any, X_seq: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Generate binary class predictions."""
        probas = DLPredictor.predict_proba(model, X_seq)
        return (probas >= threshold).astype(int)

    @staticmethod
    def predict_signal(
        model: Any,
        X_seq: np.ndarray,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
    ) -> pd.Series:
        """Convert neural model probabilities into quantitative trading signals (+1, 0, -1).

        Args:
            model: Trained neural network estimator.
            X_seq: 3D feature sequence tensor.
            long_threshold: Probability threshold to trigger Long (+1) signal.
            short_threshold: Probability threshold to trigger Short (-1) signal.

        Returns:
            pandas Series of +1, 0, -1 signal values.
        """
        probas = DLPredictor.predict_proba(model, X_seq)
        signals = np.where(
            probas >= long_threshold,
            1,
            np.where(probas <= short_threshold, -1, 0),
        )
        return pd.Series(signals, name="dl_signal")
