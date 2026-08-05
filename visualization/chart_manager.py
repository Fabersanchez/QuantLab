"""
QuantLab Chart Figure Lifecycle Manager.

Registers Matplotlib figure objects, manages multi-panel subplot grids, manages figure
lifecycles, and executes memory garbage collection.
"""

import gc
import threading
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from visualization.logger import get_visualization_logger
from visualization.themes import Theme, ThemeManager

logger = get_visualization_logger("ChartManager")


class ChartManager:
    """Institutional Figure Manager for QuantLab Graphics."""

    def __init__(self) -> None:
        """Initialize ChartManager."""
        self._figures: Dict[str, plt.Figure] = {}
        self._lock = threading.RLock()
        self._counter: int = 0

    def create_figure(
        self,
        title: str = "QuantLab Chart",
        figsize: Tuple[float, float] = (10.0, 6.0),
        theme_name: str = "dark",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a single-panel Figure and Axes.

        Returns:
            Tuple of (Figure, Axes).
        """
        with self._lock:
            fig, ax = plt.subplots(figsize=figsize)
            ThemeManager.apply(fig, theme_name=theme_name)
            ax.set_title(title)

            self._counter += 1
            fig_id = f"FIG-{self._counter:04d}"
            self._figures[fig_id] = fig
            return fig, ax

    def create_subplots(
        self,
        nrows: int,
        ncols: int,
        figsize: Tuple[float, float] = (12.0, 8.0),
        sharex: bool = True,
        theme_name: str = "dark",
    ) -> Tuple[plt.Figure, Any]:
        """Create multi-panel subplot grid layout.

        Returns:
            Tuple of (Figure, AxesArray).
        """
        with self._lock:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)
            ThemeManager.apply(fig, theme_name=theme_name)

            self._counter += 1
            fig_id = f"FIG-{self._counter:04d}"
            self._figures[fig_id] = fig
            return fig, axes

    def register_figure(self, fig: plt.Figure, name: str = "CustomChart") -> str:
        """Register an existing Matplotlib Figure."""
        with self._lock:
            self._counter += 1
            fig_id = f"FIG-{self._counter:04d}-{name}"
            self._figures[fig_id] = fig
            return fig_id

    def close_figure(self, fig_or_id: Any) -> None:
        """Close and release resources for a specific Figure instance or ID."""
        with self._lock:
            if isinstance(fig_or_id, str) and fig_or_id in self._figures:
                fig = self._figures.pop(fig_or_id)
                plt.close(fig)
            elif isinstance(fig_or_id, plt.Figure):
                plt.close(fig_or_id)
                # Remove from tracking
                keys_to_del = [k for k, v in self._figures.items() if v == fig_or_id]
                for k in keys_to_del:
                    del self._figures[k]

    def close_all(self) -> None:
        """Close all tracked figures and release memory."""
        with self._lock:
            for fig in list(self._figures.values()):
                try:
                    plt.close(fig)
                except Exception:
                    pass
            self._figures.clear()
            plt.close("all")
            gc.collect()
            logger.info("Closed all active Matplotlib figures.")
