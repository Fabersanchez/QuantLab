"""
QuantLab General Heatmap & Surface Renderer.

Renders 2D/3D parameter interaction heatmaps, calendar monthly returns heatmaps,
and custom multi-dimensional matrix heatmaps.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class HeatmapRenderer:
    """Institutional Heatmap Graphics Renderer."""

    @staticmethod
    def render_parameter_heatmap(
        df_grid: pd.DataFrame,
        param_x: str,
        param_y: str,
        metric_target: str = "fitness_score",
        title: str = "Parameter Surface Heatmap",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (9.0, 6.0),
    ) -> plt.Figure:
        """Render 2D parameter surface heatmap for two strategy parameters against target metric.

        Args:
            df_grid: Flat DataFrame containing parameters and metric columns.
            param_x: X-axis parameter name.
            param_y: Y-axis parameter name.
            metric_target: Z-axis target metric to map.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        if param_x in df_grid.columns and param_y in df_grid.columns and metric_target in df_grid.columns:
            pivot = df_grid.pivot_table(index=param_y, columns=param_x, values=metric_target, aggfunc="mean")
            im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f"{c:.2f}" if isinstance(c, float) else str(c) for c in pivot.columns])
            ax.set_yticklabels([f"{r:.2f}" if isinstance(r, float) else str(r) for r in pivot.index])

            fig.colorbar(im, ax=ax, label=metric_target)

        ax.set_title(title)
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        return fig
