"""
QuantLab Candlestick & Price Action Renderer.

Renders high-performance price action charts: OHLC Candlesticks, Heikin Ashi,
Renko Bricks, Volume Bars, Tick Charts, and Range Bars.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class CandlestickRenderer:
    """Institutional Candlestick & Bar Chart Renderer."""

    @staticmethod
    def render_ohlc(
        df: pd.DataFrame,
        title: str = "OHLC Candlestick Chart",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 7.0),
        volume_overlay: bool = True,
    ) -> plt.Figure:
        """Render standard OHLC Candlesticks with optional volume subplot.

        Args:
            df: DataFrame containing 'open', 'high', 'low', 'close', and optional 'volume'.
            title: Chart title.
            theme_name: Theme identifier.
            figsize: Figure dimensions.
            volume_overlay: Whether to include volume subplot.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)

        if volume_overlay and "volume" in df.columns:
            fig, (ax_main, ax_vol) = plt.subplots(
                2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
            )
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=figsize)
            ax_vol = None

        ThemeManager.apply(fig, theme_name=theme_name)
        ax_main.set_title(title)

        n = len(df)
        indices = np.arange(n)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        is_bull = closes >= opens

        # Render wicks (high-low lines)
        ax_main.vlines(
            indices[is_bull],
            lows[is_bull],
            highs[is_bull],
            color=theme.bull_color,
            linewidth=1.0,
            alpha=0.8,
        )
        ax_main.vlines(
            indices[~is_bull],
            lows[~is_bull],
            highs[~is_bull],
            color=theme.bear_color,
            linewidth=1.0,
            alpha=0.8,
        )

        # Render candle bodies
        body_bottom = np.minimum(opens, closes)
        body_height = np.abs(closes - opens)
        body_height = np.maximum(body_height, (highs - lows) * 0.001)  # Ensure non-zero height

        width = 0.6
        for i in range(n):
            c = theme.bull_color if is_bull[i] else theme.bear_color
            rect = Rectangle(
                (i - width / 2.0, body_bottom[i]),
                width,
                body_height[i],
                facecolor=c,
                edgecolor=c,
                linewidth=0.8,
            )
            ax_main.add_patch(rect)

        ax_main.autoscale_view()
        ax_main.set_ylabel("Price")

        # Volume bars subplot
        if ax_vol is not None:
            vols = df["volume"].values
            colors = [theme.bull_color if is_bull[i] else theme.bear_color for i in range(n)]
            ax_vol.bar(indices, vols, color=colors, width=0.6, alpha=0.7)
            ax_vol.set_ylabel("Volume")
            ax_vol.set_xlabel("Bar Index")

        return fig

    @staticmethod
    def render_heikin_ashi(
        df: pd.DataFrame,
        title: str = "Heikin Ashi Chart",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render transformed Heikin Ashi candlesticks.

        Args:
            df: DataFrame containing 'open', 'high', 'low', 'close'.
            title: Chart title.
            theme_name: Theme identifier.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure instance.
        """
        ha_df = pd.DataFrame(index=df.index)
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

        ha_open = np.zeros(len(df))
        ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2.0

        ha_df["open"] = ha_open
        ha_df["close"] = ha_close
        ha_df["high"] = np.maximum(df["high"], np.maximum(ha_open, ha_close))
        ha_df["low"] = np.minimum(df["low"], np.minimum(ha_open, ha_close))

        if "volume" in df.columns:
            ha_df["volume"] = df["volume"]

        return CandlestickRenderer.render_ohlc(
            ha_df, title=title, theme_name=theme_name, figsize=figsize, volume_overlay=False
        )

    @staticmethod
    def render_renko(
        df: pd.DataFrame,
        brick_size: Optional[float] = None,
        title: str = "Renko Chart",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render Renko brick trend blocks.

        Args:
            df: DataFrame with 'close' price column.
            brick_size: Size of ATR / fixed price Renko brick.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        closes = df["close"].values
        if brick_size is None or brick_size <= 0:
            brick_size = float(np.std(np.diff(closes))) or 1.0

        bricks = []
        current_price = closes[0]

        for p in closes:
            diff = p - current_price
            num_bricks = int(diff // brick_size) if diff > 0 else int(diff // brick_size)
            if num_bricks != 0:
                for _ in range(abs(num_bricks)):
                    direction = 1 if num_bricks > 0 else -1
                    next_price = current_price + direction * brick_size
                    bricks.append((current_price, next_price, direction))
                    current_price = next_price

        ax.set_title(f"{title} (Brick Size: {brick_size:.4f})")
        width = 0.8
        for i, (b_open, b_close, b_dir) in enumerate(bricks):
            c = theme.bull_color if b_dir > 0 else theme.bear_color
            bottom = min(b_open, b_close)
            height = abs(b_close - b_open)
            rect = Rectangle((i - width / 2.0, bottom), width, height, facecolor=c, edgecolor=c)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.set_ylabel("Price")
        ax.set_xlabel("Renko Brick Index")
        return fig
