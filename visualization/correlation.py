"""
QuantLab Cross-Asset Correlation & Covariance Renderer.

Renders Pearson, Spearman, and Kendall rank correlation matrices, covariance heatmaps,
and multi-asset pairwise scatter matrix grids.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class CorrelationRenderer:
    """Institutional Cross-Asset Correlation & Matrix Renderer."""

    @staticmethod
    def render_correlation_matrix(
        returns_df: pd.DataFrame,
        method: str = "pearson",  # 'pearson', 'spearman', 'kendall'
        title: Optional[str] = None,
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 7.0),
    ) -> plt.Figure:
        """Render correlation matrix heatmap for multi-asset returns.

        Args:
            returns_df: DataFrame where each column represents an asset returns series.
            method: Method string ('pearson', 'spearman', 'kendall').
            title: Optional chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        corr_matrix = returns_df.corr(method=method)
        im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")

        labels = corr_matrix.columns
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        # Annotate correlation numbers
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_matrix.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=9)

        ax.set_title(title or f"{method.capitalize()} Correlation Matrix")
        fig.colorbar(im, ax=ax, label="Correlation Coeff")
        return fig

    @staticmethod
    def render_covariance_matrix(
        returns_df: pd.DataFrame,
        title: str = "Covariance Matrix",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 7.0),
    ) -> plt.Figure:
        """Render asset covariance matrix heatmap.

        Args:
            returns_df: DataFrame of asset returns.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        cov_matrix = returns_df.cov()
        im = ax.imshow(cov_matrix.values, cmap="viridis", aspect="auto")

        labels = cov_matrix.columns
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Covariance")
        return fig
