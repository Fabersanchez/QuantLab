"""
QuantLab Feature Selection Framework.

Provides Variance Threshold, SelectKBest, RFE/RFECV, Mutual Information,
LASSO (L1 penalty), Boruta, and SHAP feature selection algorithms.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV


class FeatureSelector:
    """Institutional Machine Learning Feature Selection Engine."""

    @staticmethod
    def select_variance_threshold(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Select features exceeding variance threshold."""
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X.fillna(0.0))
        selected_mask = vt.get_support()
        return list(X.columns[selected_mask])

    @staticmethod
    def select_k_best(
        X: pd.DataFrame, y: pd.Series, k: int = 10, score_func: str = "f_classif"
    ) -> List[str]:
        """Select K best features using statistical tests (F-score or Mutual Information)."""
        func = f_classif if score_func == "f_classif" else mutual_info_classif
        k_num = min(k, X.shape[1])
        skb = SelectKBest(score_func=func, k=k_num)
        skb.fit(X.fillna(0.0), y)
        selected_mask = skb.get_support()
        return list(X.columns[selected_mask])

    @staticmethod
    def select_rfe(
        X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 10
    ) -> List[str]:
        """Select features via Recursive Feature Elimination (RFE) using Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        n_select = min(n_features_to_select, X.shape[1])
        rfe = RFE(estimator=model, n_features_to_select=n_select)
        rfe.fit(X.fillna(0.0), y)
        return list(X.columns[rfe.support_])

    @staticmethod
    def select_lasso(X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select non-zero coefficient features via L1 LASSO regularization."""
        lasso = LassoCV(cv=3, random_state=42, max_iter=2000)
        lasso.fit(X.fillna(0.0), y)
        coef = np.abs(lasso.coef_)
        selected_mask = coef > 1e-4
        if not any(selected_mask):
            return list(X.columns)
        return list(X.columns[selected_mask])

    @staticmethod
    def select_boruta_shadow(X: pd.DataFrame, y: pd.Series, max_iter: int = 20) -> List[str]:
        """Select features using Boruta shadow feature permutation heuristic."""
        from sklearn.ensemble import RandomForestClassifier

        X_df = X.fillna(0.0).copy()
        features = list(X_df.columns)
        selected = []

        for _ in range(max_iter):
            # Create shadow features
            shadow = X_df.apply(np.random.permutation)
            shadow.columns = [f"shadow_{c}" for c in X_df.columns]

            X_combined = pd.concat([X_df, shadow], axis=1)
            rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            rf.fit(X_combined, y)

            importances = rf.feature_importances_
            real_imp = importances[: len(features)]
            shadow_imp = importances[len(features) :]

            max_shadow = np.max(shadow_imp) if len(shadow_imp) > 0 else 0.0
            important_mask = real_imp > max_shadow
            selected.extend(X_df.columns[important_mask].tolist())

        unique_selected = list(set(selected))
        return unique_selected if unique_selected else list(X.columns)
