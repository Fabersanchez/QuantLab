"""
QuantLab Deep Learning Evaluator.

Evaluates 3D temporal sequence neural models, computes confusion matrices, ROC/PR curves,
and quantitative classification/regression performance metrics.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from deep_learning.dataloader import DLDataLoader


@dataclass
class DLEvaluationReport:
    """Dataclass holding complete Deep Learning evaluation outputs."""

    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]
    roc_curve: Dict[str, np.ndarray]
    pr_curve: Dict[str, np.ndarray]


class DLEvaluator:
    """Institutional Deep Learning Model Evaluator."""

    @staticmethod
    def evaluate(model: Any, data_loader: DLDataLoader) -> DLEvaluationReport:
        """Evaluate neural model on data_loader sequence batches.

        Args:
            model: Trained PyTorch nn.Module or fallback model.
            data_loader: DLDataLoader instance.

        Returns:
            DLEvaluationReport dataclass.
        """
        all_preds = []
        all_probs = []
        all_targets = []

        try:
            import torch
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    for X_b, y_b in data_loader:
                        if y_b is None:
                            continue
                        X_t = torch.tensor(X_b, dtype=torch.float32)
                        out = model(X_t)
                        prob = torch.sigmoid(out).cpu().numpy().ravel()
                        all_probs.extend(prob)
                        all_preds.extend((prob >= 0.5).astype(int))
                        all_targets.extend(y_b)
        except Exception:
            for X_b, y_b in data_loader:
                if y_b is None:
                    continue
                prob = model(X_b).ravel()
                all_probs.extend(prob)
                all_preds.extend((prob >= 0.5).astype(int))
                all_targets.extend(y_b)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Convert continuous floats to binary class labels if needed
        if y_true.dtype.kind == "f" and not np.all(np.isin(y_true, [0.0, 1.0, -1.0])):
            y_true = (y_true >= np.median(y_true)).astype(int)
        else:
            y_true = y_true.astype(int)

        if len(y_true) == 0:
            return DLEvaluationReport(
                metrics={"accuracy": 0.0, "roc_auc": 0.5},
                confusion_matrix=np.zeros((2, 2)),
                classification_report={},
                roc_curve={},
                pr_curve={},
            )

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

        cm = confusion_matrix(y_true, y_pred)
        cls_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        roc_data = {}
        pr_data = {}
        try:
            fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
            prec_c, rec_c, pr_thresh = precision_recall_curve(y_true, y_prob)
            roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresh}
            pr_data = {"precision": prec_c, "recall": rec_c, "thresholds": pr_thresh}
        except Exception:
            pass

        return DLEvaluationReport(
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=cls_rep,
            roc_curve=roc_data,
            pr_curve=pr_data,
        )
