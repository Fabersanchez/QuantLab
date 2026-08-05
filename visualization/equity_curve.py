"""
QuantLab Equity Curve & Portfolio Performance Renderer.

Renders equity curves, balance curves, cumulative profit, underwater drawdowns,
monthly returns heatmaps, daily return distributions, and trade timeline execution markers.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class EquityCurveRenderer:
    """Institutional Equity Curve & Performance Graphics Renderer."""

    @staticmethod
    def render_equity_curve(
        equity_series: pd.Series,
        benchmark_series: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render equity curve against optional benchmark asset series.

        Args:
            equity_series: Series of portfolio equity over time.
            benchmark_series: Optional benchmark equity series.
            title: Chart title.
            theme_name: Applied theme identifier.
            figsize: Figure size tuple.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        ax.plot(equity_series.index, equity_series.values, label="Strategy Equity", color=theme.primary_color, linewidth=2.0)

        if benchmark_series is not None and not benchmark_series.empty:
            ax.plot(
                benchmark_series.index,
                benchmark_series.values,
                label="Benchmark",
                color=theme.secondary_color,
                linestyle="--",
                linewidth=1.5,
            )

        ax.set_title(title)
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_xlabel("Time / Bar Index")
        ax.legend()
        return fig

    @staticmethod
    def render_trade_timeline(
        price_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        title: str = "Trade Execution Timeline",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 7.0),
    ) -> plt.Figure:
        """Render price series overlaid with trade entry/exit BUY and SELL markers.

        Args:
            price_df: DataFrame containing price data ('close').
            trades_df: DataFrame containing trade logs ('entry_idx', 'exit_idx', 'side', 'entry_price', 'exit_price', 'pnl').
            title: Chart title.
            theme_name: Applied theme name.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        closes = price_df["close"].values
        indices = np.arange(len(closes))
        ax.plot(indices, closes, label="Asset Price", color=theme.text_color, alpha=0.5, linewidth=1.2)

        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                entry = int(row.get("entry_idx", 0))
                exit_idx = int(row.get("exit_idx", entry + 1))
                side = str(row.get("side", "BUY")).upper()
                pnl = float(row.get("pnl", 0.0))

                entry_p = float(row.get("entry_price", closes[min(entry, len(closes) - 1)]))
                exit_p = float(row.get("exit_price", closes[min(exit_idx, len(closes) - 1)]))

                c = theme.bull_color if pnl >= 0 else theme.bear_color
                marker = "^" if "BUY" in side else "v"

                if entry < len(indices):
                    ax.scatter(entry, entry_p, color=theme.bull_color, marker=marker, s=80, zorder=5)
                if exit_idx < len(indices):
                    ax.scatter(exit_idx, exit_p, color=c, marker="o", s=60, zorder=5)
                    ax.plot([entry, exit_idx], [entry_p, exit_p], color=c, linestyle=":", alpha=0.7)

        ax.set_title(title)
        ax.set_ylabel("Price")
        ax.set_xlabel("Bar Index")
        return fig

    @staticmethod
    def render_monthly_returns(
        monthly_returns_df: pd.DataFrame,
        title: str = "Monthly Returns Heatmap",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (10.0, 5.0),
    ) -> plt.Figure:
        """Render monthly returns matrix heatmap.

        Args:
            monthly_returns_df: DataFrame indexed by Year with Month columns.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure instance.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        im = ax.imshow(monthly_returns_df.values, cmap="RdYlGn", aspect="auto")

        ax.set_xticks(np.arange(len(monthly_returns_df.columns)))
        ax.set_yticks(np.arange(len(monthly_returns_df.index)))
        ax.set_xticklabels(monthly_returns_df.columns)
        ax.set_yticklabels(monthly_returns_df.index)

        # Annotate text
        for i in range(len(monthly_returns_df.index)):
            for j in range(len(monthly_returns_df.columns)):
                val = monthly_returns_df.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="black", fontsize=8)

        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Return %")
        return fig
