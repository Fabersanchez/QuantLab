"""
QuantLab Machine Learning & Deep Learning Visualization Renderer.

Renders ML/DL performance graphics: Feature Importance, Permutation Importance, SHAP values,
Confusion Matrices, ROC Curves, Precision-Recall Curves, and Training Loss/Accuracy Curves.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class FeatureImportanceRenderer:
    """Institutional Machine Learning & Deep Learning Graphics Renderer."""

    @staticmethod
    def render_feature_importance(
        importance_dict: Dict[str, float],
        title: str = "ML Feature Importance",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 5.0),
    ) -> plt.Figure:
        """Render Feature Importance horizontal bar chart.

        Args:
            importance_dict: Dictionary mapping feature names to importance scores.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=False)
        features = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        ax.barh(features, scores, color=theme.primary_color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Importance Score")
        return fig

    @staticmethod
    def render_confusion_matrix(
        matrix: np.ndarray,
        class_labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (6.0, 5.5),
    ) -> plt.Figure:
        """Render Confusion Matrix heatmap.

        Args:
            matrix: 2D NumPy array matrix of shape (N, N).
            class_labels: Optional class label strings.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")

        labels = class_labels or [f"Class {i}" for i in range(len(matrix))]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Annotate counts inside cells
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=11)

        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        fig.colorbar(im, ax=ax)
        return fig

    @staticmethod
    def render_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float = 0.0,
        title: str = "Receiver Operating Characteristic (ROC)",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (7.0, 5.5),
    ) -> plt.Figure:
        """Render ROC Curve with AUC score.

        Args:
            fpr: False positive rate array.
            tpr: True positive rate array.
            auc_score: Area Under Curve float.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        ax.plot(fpr, tpr, color=theme.primary_color, linewidth=2.0, label=f"ROC Curve (AUC = {auc_score:.3f})")
        ax.plot([0, 1], [0, 1], color=theme.text_color, linestyle="--", alpha=0.5, label="Random Classifier")

        ax.set_title(title)
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend()
        return fig

    @staticmethod
    def render_learning_curve(
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        title: str = "Model Training & Validation Loss Curve",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 4.5),
    ) -> plt.Figure:
        """Render Training vs Validation Loss learning curve over epochs.

        Args:
            train_loss: List of training loss values per epoch.
            val_loss: Optional list of validation loss values per epoch.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        epochs = np.arange(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label="Training Loss", color=theme.primary_color, linewidth=2.0)

        if val_loss:
            ax.plot(epochs[: len(val_loss)], val_loss, label="Validation Loss", color=theme.secondary_color, linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        return fig
