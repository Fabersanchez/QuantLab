"""
QuantLab Walk Forward Optimization & Validation Renderer.

Renders train/test window timelines, in-sample vs out-of-sample window performance comparison,
and walk forward efficiency score distributions.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class WalkForwardRenderer:
    """Institutional Walk Forward Graphics Renderer."""

    @staticmethod
    def render_window_timeline(
        windows: List[Dict[str, Any]],
        title: str = "Walk Forward Window Timeline Split",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 5.0),
    ) -> plt.Figure:
        """Render In-Sample and Out-of-Sample window split timeline diagram.

        Args:
            windows: List of window dicts containing 'train_start', 'train_end', 'val_start', 'val_end'.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        height = 0.5
        for idx, win in enumerate(windows):
            tr_start = float(win.get("train_start", idx * 50))
            tr_end = float(win.get("train_end", tr_start + 100))
            val_start = float(win.get("val_start", tr_end))
            val_end = float(win.get("val_end", val_start + 30))

            # Train (In-Sample) block
            rect_tr = Rectangle(
                (tr_start, idx - height / 2.0),
                tr_end - tr_start,
                height,
                facecolor=theme.primary_color,
                edgecolor=theme.paper_color,
                alpha=0.8,
            )
            ax.add_patch(rect_tr)

            # Test (Out-of-Sample) block
            rect_val = Rectangle(
                (val_start, idx - height / 2.0),
                val_end - val_start,
                height,
                facecolor=theme.secondary_color,
                edgecolor=theme.paper_color,
                alpha=0.8,
            )
            ax.add_patch(rect_val)

        ax.set_title(title)
        ax.set_xlabel("Bar Index / Time")
        ax.set_ylabel("Window Step")
        ax.set_yticks(np.arange(len(windows)))
        ax.set_yticklabels([f"Window {i+1}" for i in range(len(windows))])
        ax.autoscale_view()
        return fig

    @staticmethod
    def render_is_vs_oos_performance(
        is_scores: List[float],
        oos_scores: List[float],
        metric_name: str = "Sharpe Ratio",
        title: str = "In-Sample vs Out-of-Sample Window Performance",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (10.0, 5.0),
    ) -> plt.Figure:
        """Render In-Sample vs Out-of-Sample performance comparison bar chart across windows.

        Args:
            is_scores: List of In-Sample metric scores per window.
            oos_scores: List of Out-of-Sample metric scores per window.
            metric_name: Target metric label.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        n = len(is_scores)
        indices = np.arange(n)
        width = 0.35

        ax.bar(indices - width / 2.0, is_scores, width, label="In-Sample (IS)", color=theme.primary_color, alpha=0.8)
        ax.bar(indices + width / 2.0, oos_scores, width, label="Out-of-Sample (OOS)", color=theme.secondary_color, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel("Window Index")
        ax.set_ylabel(metric_name)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"W{i+1}" for i in range(n)])
        ax.legend()
        return fig
