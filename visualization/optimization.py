"""
QuantLab Optimization & Parameter Space Visualization Renderer.

Renders 2D/3D parameter fitness surfaces, fitness convergence evolution (best, avg, worst),
optimization iteration history scatter plots, and multi-objective Pareto front charts.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.themes import Theme, ThemeManager


class OptimizationRenderer:
    """Institutional Optimization Graphics Renderer."""

    @staticmethod
    def render_convergence(
        convergence_curve: List[Dict[str, Any]],
        title: str = "Optimization Fitness Convergence Evolution",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (9.0, 5.0),
    ) -> plt.Figure:
        """Render fitness convergence evolution over iterations.

        Args:
            convergence_curve: List of dicts containing 'iteration', 'fitness', 'best_so_far'.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        df = pd.DataFrame(convergence_curve)
        if not df.empty:
            if "fitness" in df.columns:
                ax.plot(df["iteration"], df["fitness"], label="Iteration Fitness", color=theme.text_color, alpha=0.3, linestyle="--")
            if "best_so_far" in df.columns:
                ax.plot(df["iteration"], df["best_so_far"], label="Best Fitness So Far", color=theme.primary_color, linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel("Iteration / Evaluation Index")
        ax.set_ylabel("Fitness Score")
        ax.legend()
        return fig

    @staticmethod
    def render_pareto_front(
        objective_1_values: List[float],
        objective_2_values: List[float],
        obj_1_name: str = "Sharpe Ratio",
        obj_2_name: str = "Max Drawdown (%)",
        title: str = "Multi-Objective Pareto Front Analysis",
        theme_name: str = "dark",
        figsize: Tuple[float, float] = (8.0, 6.0),
    ) -> plt.Figure:
        """Render multi-objective trade-off Pareto front scatter plot.

        Args:
            objective_1_values: List of objective 1 values.
            objective_2_values: List of objective 2 values.
            obj_1_name: Label for objective 1.
            obj_2_name: Label for objective 2.
            title: Chart title.
            theme_name: Theme name.
            figsize: Figure size.

        Returns:
            Matplotlib Figure instance.
        """
        theme = ThemeManager.get_theme(theme_name)
        fig, ax = plt.subplots(figsize=figsize)
        ThemeManager.apply(fig, theme_name=theme_name)

        ax.scatter(objective_1_values, objective_2_values, color=theme.primary_color, alpha=0.6, s=40, label="Candidates")

        ax.set_title(title)
        ax.set_xlabel(obj_1_name)
        ax.set_ylabel(obj_2_name)
        ax.legend()
        return fig
