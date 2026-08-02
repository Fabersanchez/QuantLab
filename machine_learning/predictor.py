"""
QuantLab Inference Engine.

Generates model predictions, class probabilities, and converts ML inferences
into quantitative trading signals (-1 Short, 0 Neutral, +1 Long).
"""

from typing import Any, Optional
import numpy as np
import pandas as pd


class Predictor:
    """Institutional Inference Engine."""

    @staticmethod
    def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate class or continuous predictions."""
        X_clean = X.fillna(0.0)
        return model.predict(X_clean)

    @staticmethod
    def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities if model supports predict_proba."""
        X_clean = X.fillna(0.0)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_clean)
        else:
            preds = model.predict(X_clean)
            return np.column_stack([1.0 - preds, preds])

    @staticmethod
    def predict_signal(
        model: Any,
        X: pd.DataFrame,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
    ) -> pd.Series:
        """Convert ML model probability inferences into quantitative trading signals (+1, 0, -1).

        Args:
            model: Trained ML model estimator.
            X: Feature DataFrame.
            long_threshold: Probability threshold to trigger Long (+1) signal.
            short_threshold: Probability threshold to trigger Short (-1) signal.

        Returns:
            pandas Series of +1, 0, -1 signal values.
        """
        probas = Predictor.predict_proba(model, X)

        # For binary classification (col 1 = positive class)
        if probas.ndim == 2 and probas.shape[1] >= 2:
            p_long = probas[:, 1]
        else:
            p_long = probas.ravel()

        signals = np.where(
            p_long >= long_threshold,
            1,
            np.where(p_long <= short_threshold, -1, 0),
        )

        return pd.Series(signals, index=X.index, name="ml_signal")
