"""
QuantLab Feature Importance Evaluator.

Evaluates feature significance, generating feature rankings, scores, importance
metrics, contributions, and stability summaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class FeatureImportanceReport:
    """Importance and stability report for predictive features.

    Attributes:
        rankings: List of tuple (rank, feature_name, score).
        scores: Dictionary mapping feature name to numerical importance score.
        contribution_map: Percentage contribution of each feature (sums to 100%).
        top_features: Top K most influential features.
    """

    rankings: List[tuple] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    contribution_map: Dict[str, float] = field(default_factory=dict)
    top_features: List[str] = field(default_factory=list)


class FeatureImportanceEvaluator:
    """Evaluates and ranks predictive feature significance."""

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k: int = 10,
    ) -> FeatureImportanceReport:
        """Evaluate feature importance using target correlation magnitude.

        Args:
            X: Input feature matrix DataFrame.
            y: Target variable Series.
            top_k: Number of top features to isolate in top_features.

        Returns:
            FeatureImportanceReport object.
        """
        scores: Dict[str, float] = {}
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0)

        for col in numeric_X.columns:
            corr = float(np.abs(numeric_X[col].corr(y)))
            scores[col] = 0.0 if np.isnan(corr) else corr

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = [
            (idx + 1, col, score) for idx, (col, score) in enumerate(sorted_items)
        ]

        total_score = sum(scores.values()) or 1e-8
        contrib_map = {
            col: round((score / total_score) * 100.0, 2)
            for col, score in sorted_items
        }

        top_feats = [col for col, _ in sorted_items[:top_k]]

        return FeatureImportanceReport(
            rankings=rankings,
            scores=scores,
            contribution_map=contrib_map,
            top_features=top_feats,
        )
