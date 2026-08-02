"""
QuantLab Machine Learning Model Evaluator.

Evaluates model performance against validation/test sets, computes confusion matrices,
classification reports, ROC/PR curves, and compares candidate models against baseline strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve

from machine_learning.metrics import MLMetricsCalculator


@dataclass
class EvaluationReport:
    """Dataclass holding complete model evaluation outputs."""

    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    roc_curve: Dict[str, np.ndarray]
    pr_curve: Dict[str, np.ndarray]


class ModelEvaluator:
    """Institutional Machine Learning Model Evaluator."""

    @staticmethod
    def evaluate(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationReport:
        """Evaluate model on Out-of-Sample test set.

        Args:
            model: Trained estimator.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            EvaluationReport object.
        """
        X_clean = X_test.fillna(0.0)
        y_pred = model.predict(X_clean)

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_clean)
            except Exception:
                y_prob = None

        # Metrics
        metrics = MLMetricsCalculator.calculate_classification_metrics(y_test, y_pred, y_prob)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification Report
        cls_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # ROC & PR Curves
        roc_data = {}
        pr_data = {}

        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    prob_1d = y_prob[:, 1]
                else:
                    prob_1d = y_prob

                fpr, tpr, roc_thresh = roc_curve(y_test, prob_1d)
                prec, rec, pr_thresh = precision_recall_curve(y_test, prob_1d)

                roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresh}
                pr_data = {"precision": prec, "recall": rec, "thresholds": pr_thresh}
            except Exception:
                pass

        return EvaluationReport(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=cls_rep,
            roc_curve=roc_data,
            pr_curve=pr_data,
        )
