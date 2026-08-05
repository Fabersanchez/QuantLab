"""
QuantLab Financial Statistics & Rolling Metrics Renderer.

Renders statistical distributions: returns histogram with fitted probability density,
rolling Sharpe ratio, rolling Sortino ratio, rolling volatility, and rolling drawdown.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from visualization.themes import Theme, ThemeManager


class StatisticsRenderer:
    """Institutional Financial Statistics Renderer."""

    @staticmethod
    def render_returns_histogram(
        returns_series: pd.Series,
        fit_distribution: bool = True,
        title: str = "Daily Returns Distribution & Fitted Density",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 5.0),
    ) -> plt.Figure:
        """Render returns histogram with optional fitted Gaussian distribution density curve.

        Args:
            returns_series: Returns percentage series.
            fit_distribution: Whether to overlay fitted Normal density curve.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        clean_returns = returns_series.dropna().values
        if len(clean_returns) > 0:
            count, bins, _ = ax.hist(
                clean_returns, bins=30, density=True, color=theme.primary_color, alpha=0.6, edgecolor=theme.paper_color
            )

            if fit_distribution and len(clean_returns) > 2:
                mu, std = float(np.mean(clean_returns)), float(np.std(clean_returns))
                if std > 0:
                    y = stats.norm.pdf(bins, mu, std)
                    ax.plot(bins, y, color=theme.secondary_color, linewidth=2.0, label=f"Normal Fit (μ={mu:.4f}, σ={std:.4f})")

        ax.set_title(title)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()
        return fig

    @staticmethod
    def render_rolling_sharpe(
        returns_series: pd.Series,
        window: int = 63,
        title: str = "Rolling Sharpe Ratio",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (10.0, 4.5),
    ) -> plt.Figure:
        """Render rolling Sharpe ratio over time.

        Args:
            returns_series: Daily/period return series.
            window: Rolling window size in bars.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        rolling_mean = returns_series.rolling(window).mean()
        rolling_std = returns_series.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=theme.primary_color, linewidth=1.8, label=f"Rolling {window}-Bar Sharpe")
        ax.axhline(0, color=theme.text_color, linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_xlabel("Time / Bar Index")
        ax.legend()
        return fig
