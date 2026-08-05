"""
QuantLab Financial Animation & Dynamic Replay Engine.

Renders dynamic animated candlestick replays, equity growth animations, and optimization swarm
evolution frames, supporting export to GIF, MP4, and HTML5 video formats using Matplotlib FuncAnimation.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.candlestick import CandlestickRenderer
from visualization.logger import get_visualization_logger
from visualization.themes import ThemeManager

logger = get_visualization_logger("AnimationEngine")


class AnimationEngine:
    """Institutional Dynamic Graphic Animation Engine."""

    @staticmethod
    def create_candlestick_replay_animation(
        df: pd.DataFrame,
        interval_ms: int = 100,
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (10.0, 5.0),
    ) -> animation.FuncAnimation:
        """Create bar-by-bar animated candlestick replay animation.

        Args:
            df: DataFrame containing OHLC data.
            interval_ms: Delay between frames in milliseconds.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib FuncAnimation object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        n = len(df)

        def update(frame: int) -> None:
            ax.clear()
            ThemeManager.apply(fig, theme_name=theme_name)
            sub_df = df.iloc[: frame + 1]
            ax.plot(sub_df.index, sub_df["close"], color="#2962ff", linewidth=1.5)
            ax.set_title(f"Market Replay - Bar {frame + 1} / {n}")
            ax.set_ylabel("Price")

        anim = animation.FuncAnimation(fig, update, frames=n, interval=interval_ms, repeat=False)
        logger.info(f"Created candlestick replay animation ({n} frames).")
        return anim

    @staticmethod
    def save_animation(
        anim: animation.FuncAnimation,
        filepath: str,
        fps: int = 15,
    ) -> str:
        """Export animation to GIF or MP4 file format.

        Args:
            anim: FuncAnimation instance.
            filepath: Destination file path (.gif or .mp4).
            fps: Frames per second.

        Returns:
            Absolute file path.
        """
        if filepath.endswith(".gif"):
            anim.save(filepath, writer="pillow", fps=fps)
        else:
            try:
                anim.save(filepath, writer="ffmpeg", fps=fps)
            except Exception:
                # Fallback to GIF if ffmpeg is missing
                gif_path = filepath if filepath.endswith(".gif") else filepath + ".gif"
                anim.save(gif_path, writer="pillow", fps=fps)
                filepath = gif_path

        logger.log_export("Animation", "GIF/MP4", filepath)
        return filepath
