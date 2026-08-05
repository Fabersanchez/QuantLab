"""
QuantLab Optimization Visualization Engine.

Generates analytical visual data plots: convergence curves, parameter heatmaps,
parameter importance bar charts, fitness distributions, population fitness evolution,
and multi-algorithm comparison charts using Matplotlib.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimization.history import OptimizationHistory


class OptimizationVisualizer:
    """Institutional Visualization Engine for Strategy Optimization Outputs."""

    @staticmethod
    def plot_convergence_curve(history: OptimizationHistory, filepath: Optional[str] = None) -> plt.Figure:
        """Plot running maximum fitness convergence curve across iterations."""
        curve = history.get_convergence_curve()
        df = pd.DataFrame(curve)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not df.empty:
            ax.plot(df["iteration"], df["fitness"], label="Iteration Fitness", alpha=0.4, color="gray", linestyle="--")
            ax.plot(df["iteration"], df["best_so_far"], label="Best Fitness So Far", color="navy", linewidth=2)

        ax.set_title("Optimization Convergence Curve")
        ax.set_xlabel("Iteration / Evaluation Index")
        ax.set_ylabel("Fitness Score")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        if filepath:
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
        return fig

    @staticmethod
    def plot_parameter_importance(history: OptimizationHistory, filepath: Optional[str] = None) -> plt.Figure:
        """Plot estimated parameter importance bar chart based on correlation with fitness."""
        df = history.to_dataframe()
        fig, ax = plt.subplots(figsize=(8, 4.5))

        param_cols = [c for c in df.columns if c.startswith("param_")]
        if param_cols and "fitness_score" in df.columns:
            corrs = {}
            for col in param_cols:
                p_name = col.replace("param_", "")
                try:
                    corr = abs(df[col].astype(float).corr(df["fitness_score"]))
                    corrs[p_name] = corr if not np.isnan(corr) else 0.0
                except Exception:
                    corrs[p_name] = 0.0

            names = list(corrs.keys())
            scores = list(corrs.values())

            ax.barh(names, scores, color="teal")
            ax.set_title("Estimated Parameter Importance (Correlation with Fitness)")
            ax.set_xlabel("Absolute Correlation Score")
            ax.grid(True, linestyle=":", alpha=0.6)

        if filepath:
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
        return fig

    @staticmethod
    def plot_fitness_distribution(history: OptimizationHistory, filepath: Optional[str] = None) -> plt.Figure:
        """Plot histogram distribution of evaluated candidate fitness scores."""
        df = history.to_dataframe()
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if "fitness_score" in df.columns and not df.empty:
            scores = df["fitness_score"].values
            ax.hist(scores, bins=15, color="darkslateblue", edgecolor="black", alpha=0.7)
            ax.set_title("Distribution of Fitness Scores")
            ax.set_xlabel("Fitness Score")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle=":", alpha=0.6)

        if filepath:
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
        return fig

    @staticmethod
    def plot_algorithm_comparison(
        histories_map: Dict[str, OptimizationHistory], filepath: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparative convergence curves across multiple algorithms."""
        fig, ax = plt.subplots(figsize=(9, 5))

        for algo_name, hist in histories_map.items():
            curve = hist.get_convergence_curve()
            df = pd.DataFrame(curve)
            if not df.empty:
                ax.plot(df["iteration"], df["best_so_far"], label=algo_name, linewidth=2)

        ax.set_title("Multi-Algorithm Optimization Convergence Comparison")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness So Far")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

        if filepath:
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
        return fig
