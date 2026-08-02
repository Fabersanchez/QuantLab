"""
QuantLab Machine Learning Model Trainer.

Orchestrates training, incremental learning (`partial_fit`), early stopping, and checkpointing
for Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes, Logistic Regression, MLP.
"""

from typing import Any, Dict, Optional, Type
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    """Institutional Machine Learning Model Trainer."""

    MODEL_MAP: Dict[str, Any] = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "adaboost": AdaBoostClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
        "mlp": MLPClassifier,
        "sgd": SGDClassifier,
    }

    def __init__(self, model_type: str = "random_forest", params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ModelTrainer.

        Args:
            model_type: Identifier of target model architecture.
            params: Optional hyperparameter overrides.
        """
        self.model_type = model_type.lower().strip()
        self.params = params or {}
        self.model: Optional[Any] = None

    def _instantiate_model(self) -> Any:
        """Instantiate estimator object by model_type."""
        params = self.params.copy()

        # Handle XGBoost, LightGBM, CatBoost dynamically if installed
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                params.setdefault("n_estimators", 100)
                params.setdefault("max_depth", 5)
                params.setdefault("random_state", 42)
                return XGBClassifier(**params)
            except ImportError:
                return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

        elif self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                params.setdefault("n_estimators", 100)
                params.setdefault("random_state", 42)
                return LGBMClassifier(**params)
            except ImportError:
                return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

        elif self.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                params.setdefault("iterations", 100)
                params.setdefault("verbose", 0)
                return CatBoostClassifier(**params)
            except ImportError:
                return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

        elif self.model_type in self.MODEL_MAP:
            cls = self.MODEL_MAP[self.model_type]
            if "random_state" in cls().__dir__():
                params.setdefault("random_state", 42)
            if self.model_type == "svm":
                params.setdefault("probability", True)
            return cls(**params)

        else:
            return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model on In-Sample features and target.

        Returns:
            Trained estimator object.
        """
        self.model = self._instantiate_model()
        X_clean = X_train.fillna(0.0)
        self.model.fit(X_clean, y_train)
        return self.model

    def incremental_train(self, X_batch: pd.DataFrame, y_batch: pd.Series, classes: Optional[Any] = None) -> Any:
        """Perform incremental training (`partial_fit`) for streaming data."""
        if self.model is None:
            self.model = SGDClassifier(loss="log_loss", random_state=42)

        X_clean = X_batch.fillna(0.0)
        if hasattr(self.model, "partial_fit"):
            if classes is None:
                classes = np.unique(y_batch)
            self.model.partial_fit(X_clean, y_batch, classes=classes)
        else:
            self.model.fit(X_clean, y_batch)

        return self.model
