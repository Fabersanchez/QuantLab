"""
QuantLab Portfolio & Asset Allocation Renderer.

Renders asset allocation pie/donut/stacked area charts, risk exposure breakdowns,
diversification ratio metrics, and return contribution waterfall charts.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class PortfolioRenderer:
    """Institutional Portfolio Graphics Renderer."""

    @staticmethod
    def render_asset_allocation(
        allocations: Dict[str, float],
        title: str = "Portfolio Asset Allocation",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (7.0, 6.0),
    ) -> plt.Figure:
        """Render asset allocation donut / pie chart.

        Args:
            allocations: Dictionary mapping asset names to weight percentage.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        labels = list(allocations.keys())
        weights = list(allocations.values())

        pie_result = ax.pie(
            weights,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor="white"),
        )

        # Handle type stubs variation for ax.pie return tuple
        if len(pie_result) == 3:
            _, texts, autotexts = pie_result
            all_text_elements = list(texts) + list(autotexts)
        else:
            _, texts = pie_result[0], pie_result[1]
            all_text_elements = list(texts)

        for text_item in all_text_elements:
            text_item.set_fontsize(9)

        ax.set_title(title)
        return fig

    @staticmethod
    def render_return_contribution(
        contributions: Dict[str, float],
        title: str = "Asset Return Contribution",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 5.0),
    ) -> plt.Figure:
        """Render asset return contribution horizontal bar chart.

        Args:
            contributions: Dictionary mapping asset name to return contribution %.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        assets = list(contributions.keys())
        vals = list(contributions.values())
        colors = [theme.bull_color if v >= 0 else theme.bear_color for v in vals]

        ax.barh(assets, vals, color=colors, alpha=0.8)
        ax.axvline(0, color=theme.text_color, linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Contribution (%)")
        return fig
