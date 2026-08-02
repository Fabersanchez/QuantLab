"""
QuantLab Hyperparameter Manager.

Provides standardized hyperparameter search space specifications for Random Forest, Extra Trees,
Gradient Boosting, XGBoost, LightGBM, CatBoost, SVM, and MLP models.
"""

from typing import Any, Dict, List


class HyperparameterManager:
    """Institutional Hyperparameter Search Space Specification Manager."""

    SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "extra_trees": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
        "xgboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 1.0],
        },
        "lightgbm": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 10],
        },
        "catboost": {
            "iterations": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "depth": [4, 6, 8],
        },
        "svm": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "logistic_regression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "mlp": {
            "hidden_layer_sizes": [(50,), (100,), (50, 25)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },
    }

    @staticmethod
    def get_search_space(model_type: str) -> Dict[str, List[Any]]:
        """Fetch pre-configured hyperparameter search space for model type."""
        m = model_type.lower().strip()
        return HyperparameterManager.SEARCH_SPACES.get(m, {})
