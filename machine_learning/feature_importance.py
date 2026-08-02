"""
QuantLab Feature Importance Analysis Engine.

Computes Mean Decrease Impurity (MDI / Gini), Mean Decrease Accuracy (MDA),
Permutation Importance, and Single Feature Importance (SFI).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:
    """Institutional Feature Importance Analyzer."""

    @staticmethod
    def calculate_tree_mdi(model: Any, feature_names: List[str]) -> pd.Series:
        """Calculate Mean Decrease Impurity (MDI / Gini importance).

        Args:
            model: Trained tree-based estimator.
            feature_names: List of feature names.

        Returns:
            pandas Series of feature importances sorted descending.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute.")

        imp = model.feature_importances_
        series = pd.Series(imp, index=feature_names, name="MDI_Importance")
        return series.sort_values(ascending=False)

    @staticmethod
    def calculate_permutation_importance(
        model: Any, X_val: pd.DataFrame, y_val: pd.Series, n_repeats: int = 10
    ) -> pd.Series:
        """Calculate Permutation Importance on validation dataset.

        Args:
            model: Trained model estimator.
            X_val: Validation features.
            y_val: Validation target labels.
            n_repeats: Number of permutation iterations.

        Returns:
            pandas Series of mean permutation importance.
        """
        res = permutation_importance(
            model, X_val.fillna(0.0), y_val, n_repeats=n_repeats, random_state=42
        )
        series = pd.Series(res.importances_mean, index=X_val.columns, name="Permutation_Importance")
        return series.sort_values(ascending=False)

    @staticmethod
    def calculate_sfi(
        model_cls: Any, X: pd.DataFrame, y: pd.Series, scoring_func: Optional[Any] = None
    ) -> pd.Series:
        """Calculate Single Feature Importance (SFI).

        Fits model individually on each single feature to eliminate multi-collinearity masking.
        """
        scores = {}
        for col in X.columns:
            try:
                x_single = X[[col]].fillna(0.0)
                m = model_cls()
                m.fit(x_single, y)
                score = m.score(x_single, y)
                scores[col] = float(score)
            except Exception:
                scores[col] = 0.0

        series = pd.Series(scores, name="SFI_Importance")
        return series.sort_values(ascending=False)
