"""
QuantLab Monte Carlo Simulation & Robustness Renderer.

Renders multi-path simulation fan charts, confidence interval bands (90%, 95%, 99%),
final equity distributions, and worst/best case stress envelope curves.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class MonteCarloRenderer:
    """Institutional Monte Carlo Simulation Graphics Renderer."""

    @staticmethod
    def render_simulation_fan_chart(
        simulation_matrix: np.ndarray,  # Shape (n_simulations, n_bars)
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
        title: str = "Monte Carlo Simulation Fan Chart",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render multi-path simulation fan chart with confidence interval bands.

        Args:
            simulation_matrix: 2D NumPy array of equity curve paths.
            confidence_levels: Confidence percentile levels.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        n_sims, n_bars = simulation_matrix.shape
        indices = np.arange(n_bars)

        # Plot sample simulation paths in background
        sample_size = min(50, n_sims)
        for i in range(sample_size):
            ax.plot(indices, simulation_matrix[i], color=theme.primary_color, alpha=0.08, linewidth=0.8)

        # Compute median and percentiles
        median_path = np.median(simulation_matrix, axis=0)
        ax.plot(indices, median_path, color=theme.primary_color, linewidth=2.5, label="Median Path")

        # Confidence bands
        alphas = [0.35, 0.25, 0.15]
        for idx, conf in enumerate(sorted(confidence_levels)):
            lower_p = (1.0 - conf) / 2.0 * 100.0
            upper_p = (1.0 - (1.0 - conf) / 2.0) * 100.0

            lower_band = np.percentile(simulation_matrix, lower_p, axis=0)
            upper_band = np.percentile(simulation_matrix, upper_p, axis=0)

            ax.fill_between(
                indices,
                lower_band,
                upper_band,
                color=theme.accent_color,
                alpha=alphas[idx % len(alphas)],
                label=f"{int(conf*100)}% CI Band",
            )

        ax.set_title(title)
        ax.set_ylabel("Portfolio Equity ($)")
        ax.set_xlabel("Time / Bar Index")
        ax.legend()
        return fig

    @staticmethod
    def render_risk_distribution(
        final_equities: np.ndarray,
        initial_capital: float = 100000.0,
        title: str = "Monte Carlo Final Equity Distribution",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (9.0, 5.0),
    ) -> plt.Figure:
        """Render final equity & risk distribution histogram.

        Args:
            final_equities: 1D NumPy array of final equity outcomes.
            initial_capital: Account initial capital.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        ax.hist(final_equities, bins=25, color=theme.primary_color, edgecolor=theme.paper_color, alpha=0.7)
        ax.axvline(initial_capital, color=theme.secondary_color, linestyle="--", linewidth=2.0, label="Initial Capital")

        mean_eq = float(np.mean(final_equities))
        ax.axvline(mean_eq, color=theme.bull_color, linestyle="-", linewidth=2.0, label=f"Mean: ${mean_eq:,.0f}")

        ax.set_title(title)
        ax.set_xlabel("Final Portfolio Equity ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        return fig
