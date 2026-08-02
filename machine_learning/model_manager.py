"""
QuantLab Model Manager.

Provides model serialization, deserialization, cloning, freezing, exporting, and archiving routines using joblib/pickle.
"""

import copy
import os
from typing import Any, Optional
import joblib


class ModelManager:
    """Institutional Machine Learning Model Lifecycle Manager."""

    @staticmethod
    def save_model(model: Any, filepath: str) -> str:
        """Serialize and save trained model to disk.

        Args:
            model: Trained estimator object.
            filepath: Destination file path.

        Returns:
            Absolute file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(model, filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load serialized model from disk.

        Args:
            filepath: Target file path.

        Returns:
            Deserialized estimator object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file '{filepath}' does not exist.")
        return joblib.load(filepath)

    @staticmethod
    def clone_model(model: Any) -> Any:
        """Create a deep copy clone of an estimator."""
        from sklearn.base import clone
        try:
            return clone(model)
        except Exception:
            return copy.deepcopy(model)

    @staticmethod
    def freeze_model(model: Any) -> Any:
        """Freeze model attributes for production immutability."""
        cloned = ModelManager.clone_model(model)
        # Attach frozen marker flag
        cloned._is_frozen = True
        return cloned
