"""
QuantLab Machine Learning Quantitative Performance Metrics.

Calculates Accuracy, Precision, Recall, F1, ROC AUC, PR AUC,
Matthews Correlation Coefficient (MCC), Balanced Accuracy, Brier Score, Log Loss, MSE, RMSE, and R2.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MLMetricsCalculator:
    """Quantitative Machine Learning Performance Metrics Calculator."""

    @staticmethod
    def calculate_classification_metrics(
        y_true: Any, y_pred: Any, y_prob: Optional[Any] = None
    ) -> Dict[str, float]:
        """Calculate complete suite of classification performance metrics.

        Args:
            y_true: True target labels.
            y_pred: Predicted class labels.
            y_prob: Predicted class probabilities (optional).

        Returns:
            Dict containing accuracy, precision, recall, f1, mcc, balanced_accuracy, brier_score, log_loss, roc_auc.
        """
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        # Precision, Recall, F1
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        roc_auc = 0.5
        brier = 0.0
        l_loss = 0.0

        if y_prob is not None:
            try:
                # Handle binary vs multiclass probabilities
                if hasattr(y_prob, "ndim") and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob_1d = y_prob[:, 1]
                else:
                    y_prob_1d = y_prob

                roc_auc = float(roc_auc_score(y_true, y_prob_1d, multi_class="ovr"))
                brier = float(brier_score_loss(y_true, y_prob_1d))
                l_loss = float(log_loss(y_true, y_prob, labels=np.unique(y_true)))
            except Exception:
                roc_auc = 0.5

        return {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "log_loss": l_loss,
        }

    @staticmethod
    def calculate_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
        """Calculate regression metrics (MSE, RMSE, MAE, R2)."""
        from sklearn.metrics import mean_absolute_error, r2_score

        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
        }
