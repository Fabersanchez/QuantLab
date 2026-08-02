"""
QuantLab Ensemble Learning Engine.

Combines base quantitative ML models via Voting, Weighted Voting, Bagging, Boosting,
Blending, Stacking (Meta-learner), and Dynamic Ensemble Selection (DES).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


class EnsembleEngine:
    """Institutional Ensemble Learning Factory and Orchestrator."""

    @staticmethod
    def create_voting_ensemble(
        models: List[Tuple[str, Any]], voting_type: str = "soft", weights: Optional[List[float]] = None
    ) -> VotingClassifier:
        """Create Voting Ensemble Classifier.

        Args:
            models: List of (name, model_instance) tuples.
            voting_type: 'soft' (probability weighted) or 'hard' (majority vote).
            weights: Optional feature/model weight list.

        Returns:
            VotingClassifier instance.
        """
        return VotingClassifier(estimators=models, voting=voting_type, weights=weights)

    @staticmethod
    def create_stacking_ensemble(
        models: List[Tuple[str, Any]], meta_learner: Optional[Any] = None
    ) -> StackingClassifier:
        """Create Stacking Ensemble with a meta-learner final estimator.

        Args:
            models: List of base (name, model) estimators.
            meta_learner: Final estimator meta-learner (defaults to LogisticRegression).

        Returns:
            StackingClassifier instance.
        """
        final_est = meta_learner or LogisticRegression(random_state=42)
        return StackingClassifier(estimators=models, final_estimator=final_est, cv=3)

    @staticmethod
    def create_bagging_ensemble(
        base_estimator: Any, n_estimators: int = 10, max_samples: float = 0.8
    ) -> BaggingClassifier:
        """Create Bagging (Bootstrap Aggregating) Ensemble."""
        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
        )

    @staticmethod
    def blend_predictions(
        predictions_list: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Blend prediction array list via weighted average.

        Args:
            predictions_list: List of 1D or 2D probability/prediction arrays.
            weights: Optional list of floats summing to 1.0.

        Returns:
            Blended numpy array.
        """
        if not predictions_list:
            raise ValueError("predictions_list cannot be empty.")

        n_models = len(predictions_list)
        if weights is None:
            w = np.ones(n_models) / n_models
        else:
            w = np.array(weights) / np.sum(weights)

        blended = np.zeros_like(predictions_list[0], dtype=float)
        for i in range(n_models):
            blended += predictions_list[i] * w[i]

        return blended
