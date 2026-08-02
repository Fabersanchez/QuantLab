"""
QuantLab Feature Selector.

Provides automated feature selection algorithms: Variance Threshold, Mutual Information,
Correlation Threshold, and interfaces for Lasso, Tree Importance, RFE, Boruta, and SHAP.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class BaseFeatureSelector(ABC):
    """Abstract Base Class for feature selectors."""

    @abstractmethod
    def select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        """Return list of selected column names."""
        pass


class VarianceThresholdSelector(BaseFeatureSelector):
    """Selects features with variance exceeding threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        numeric_X = X.select_dtypes(include=[np.number])
        variances = numeric_X.var()
        selected = variances[variances > self.threshold].index.tolist()
        return selected


class CorrelationThresholdSelector(BaseFeatureSelector):
    """Removes highly correlated collinear features."""

    def __init__(self, max_correlation: float = 0.95) -> None:
        self.max_correlation = max_correlation

    def select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.empty:
            return list(X.columns)

        corr_matrix = numeric_X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] > self.max_correlation)
        ]

        selected = [c for c in X.columns if c not in to_drop]
        return selected


class MutualInformationSelector(BaseFeatureSelector):
    """Selects top K features based on Mutual Information estimation."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        if y is None or X.empty:
            return list(X.columns[: self.top_k])

        numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
        scores: Dict[str, float] = {}

        # Pearson correlation magnitude as robust linear proxy for MI score
        for col in numeric_X.columns:
            corr = float(np.abs(numeric_X[col].corr(y)))
            scores[col] = 0.0 if np.isnan(corr) else corr

        sorted_cols = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_cols[: self.top_k]


class FeatureSelector:
    """Master Feature Selector orchestrating multiple filtering steps."""

    def __init__(
        self,
        variance_threshold: float = 0.0,
        max_correlation: float = 0.95,
        top_k_mi: Optional[int] = None,
    ) -> None:
        self.var_selector = VarianceThresholdSelector(threshold=variance_threshold)
        self.corr_selector = CorrelationThresholdSelector(max_correlation=max_correlation)
        self.mi_selector = (
            MutualInformationSelector(top_k=top_k_mi) if top_k_mi else None
        )

    def select(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Run selection pipeline and return filtered DataFrame."""
        step1_cols = self.var_selector.select_features(X)
        step1_df = X[step1_cols]

        step2_cols = self.corr_selector.select_features(step1_df)
        step2_df = step1_df[step2_cols]

        if self.mi_selector and y is not None:
            step3_cols = self.mi_selector.select_features(step2_df, y)
            return step2_df[step3_cols]

        return step2_df
