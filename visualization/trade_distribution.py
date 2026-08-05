"""
QuantLab Trade Distribution Graphics Renderer.

Renders trade performance analytical distributions: winning vs losing trade PnLs, trade duration
vs return scatter plots, win/loss streak histograms, and time-of-day/day-of-week returns.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class TradeDistributionRenderer:
    """Institutional Trade Distribution Graphics Renderer."""

    @staticmethod
    def render_pnl_distribution(
        trades_df: pd.DataFrame,
        title: str = "Trade PnL Distribution",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (9.0, 5.0),
    ) -> plt.Figure:
        """Render winning vs losing trades PnL histogram.

        Args:
            trades_df: DataFrame containing 'pnl'.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        if not trades_df.empty and "pnl" in trades_df.columns:
            pnls = trades_df["pnl"].values
            wins = pnls[pnls >= 0]
            losses = pnls[pnls < 0]

            if len(wins) > 0:
                ax.hist(wins, bins=15, color=theme.bull_color, alpha=0.7, label="Winning Trades")
            if len(losses) > 0:
                ax.hist(losses, bins=15, color=theme.bear_color, alpha=0.7, label="Losing Trades")

            ax.axvline(0, color=theme.text_color, linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Trade PnL ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        return fig

    @staticmethod
    def render_pnl_vs_duration(
        trades_df: pd.DataFrame,
        title: str = "Trade Duration vs Return PnL",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (9.0, 5.0),
    ) -> plt.Figure:
        """Render scatter plot comparing holding duration against trade PnL.

        Args:
            trades_df: DataFrame containing 'duration' and 'pnl'.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        if not trades_df.empty and "duration" in trades_df.columns and "pnl" in trades_df.columns:
            durations = trades_df["duration"].values
            pnls = trades_df["pnl"].values
            colors = [theme.bull_color if p >= 0 else theme.bear_color for p in pnls]

            ax.scatter(durations, pnls, c=colors, alpha=0.7, s=40, edgecolors="none")
            ax.axhline(0, color=theme.text_color, linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Holding Duration (Bars)")
        ax.set_ylabel("Trade PnL ($)")
        return fig
