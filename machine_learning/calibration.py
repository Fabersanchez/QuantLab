"""
QuantLab Probability Calibration Engine.

Calibrates uncalibrated model probabilities using Platt Scaling (Sigmoid),
Isotonic Regression, and Temperature Scaling.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    """Institutional Probability Calibration Engine."""

    @staticmethod
    def calibrate(
        model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        method: str = "sigmoid",  # 'sigmoid' (Platt) or 'isotonic'
    ) -> Any:
        """Calibrate uncalibrated model probabilities on validation set.

        Args:
            model: Pre-trained estimator.
            X_val: Validation features.
            y_val: Validation target labels.
            method: 'sigmoid' for Platt Scaling or 'isotonic' for Isotonic Regression.

        Returns:
            CalibratedClassifierCV wrapper model.
        """
        method_clean = method.lower().strip()
        try:
            calibrated = CalibratedClassifierCV(
                estimator=model, method=method_clean, cv="prefit"
            )
            calibrated.fit(X_val.fillna(0.0), y_val)
        except Exception:
            calibrated = CalibratedClassifierCV(
                estimator=model, method=method_clean, cv=3
            )
            calibrated.fit(X_val.fillna(0.0), y_val)
        return calibrated

    @staticmethod
    def temperature_scale(
        logits: np.ndarray, temperature: float = 1.5
    ) -> np.ndarray:
        """Apply Softmax Temperature Scaling to raw logit outputs."""
        if temperature <= 0:
            temperature = 1.0
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
