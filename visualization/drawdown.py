"""
QuantLab Drawdown & Underwater Analysis Renderer.

Renders underwater drawdown depth percentage charts, monetary drawdowns ($), drawdown duration
histograms, and peak-to-trough recovery shading.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class DrawdownRenderer:
    """Institutional Drawdown & Risk Graphics Renderer."""

    @staticmethod
    def render_underwater_drawdown(
        equity_series: pd.Series,
        title: str = "Underwater Drawdown Chart",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 5.0),
    ) -> plt.Figure:
        """Render underwater drawdown percentage filled area chart.

        Args:
            equity_series: Portfolio equity values.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        peak = equity_series.cummax()
        pct_drawdown = ((equity_series - peak) / peak) * 100.0

        ax.plot(pct_drawdown.index, pct_drawdown.values, color=theme.bear_color, linewidth=1.5)
        ax.fill_between(pct_drawdown.index, 0, pct_drawdown.values, color=theme.bear_color, alpha=0.3)

        ax.set_title(title)
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Time / Bar Index")
        ax.set_ylim(top=1.0)
        return fig

    @staticmethod
    def render_drawdown_duration_histogram(
        equity_series: pd.Series,
        title: str = "Drawdown Duration Distribution",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 4.5),
    ) -> plt.Figure:
        """Render histogram distribution of underwater drawdown recovery durations.

        Args:
            equity_series: Portfolio equity values.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        peak = equity_series.cummax()
        is_underwater = equity_series < peak

        durations: List[int] = []
        cur_dur = 0
        for underwater in is_underwater:
            if underwater:
                cur_dur += 1
            else:
                if cur_dur > 0:
                    durations.append(cur_dur)
                    cur_dur = 0
        if cur_dur > 0:
            durations.append(cur_dur)

        if not durations:
            durations = [0]

        ax.hist(durations, bins=15, color=theme.secondary_color, edgecolor=theme.paper_color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Drawdown Duration (Bars)")
        ax.set_ylabel("Frequency")
        return fig
