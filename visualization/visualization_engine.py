"""
QuantLab Master Visualization Engine.

Centralizes absolutely all graphical representations across QuantLab: Candlestick, Backtest Equity,
Drawdown, Statistics, Correlations, Machine Learning, Deep Learning, Reinforcement Learning,
Optimization, Monte Carlo, Walk Forward, and Portfolio allocations.
Fully independent, usable from scripts, notebooks, APIs, and GUIs.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.animation import AnimationEngine
from visualization.candlestick import CandlestickRenderer
from visualization.cache import VisualizationCache
from visualization.chart_manager import ChartManager
from visualization.correlation import CorrelationRenderer
from visualization.drawdown import DrawdownRenderer
from visualization.equity_curve import EquityCurveRenderer
from visualization.exporter import VisualizationExporter
from visualization.feature_importance import FeatureImportanceRenderer
from visualization.heatmap import HeatmapRenderer
from visualization.logger import get_visualization_logger
from visualization.montecarlo import MonteCarloRenderer
from visualization.optimization import OptimizationRenderer
from visualization.portfolio import PortfolioRenderer
from visualization.statistics import StatisticsRenderer
from visualization.themes import Theme, ThemeManager
from visualization.walkforward import WalkForwardRenderer

logger = get_visualization_logger("VisualizationEngine")


class VisualizationEngine:
    """Master Institutional Visualization Engine for QuantLab."""

    def __init__(self, theme_name: str = "dark", enable_cache: bool = True) -> None:
        """Initialize VisualizationEngine.

        Args:
            theme_name: Default theme ('dark', 'light', 'institutional', 'trading_desk', 'research').
            enable_cache: Whether to enable chart rendering cache.
        """
        self.theme_name = theme_name
        self.enable_cache = enable_cache
        self.chart_manager = ChartManager()
        self.cache = VisualizationCache()

    def set_theme(self, theme_name: str) -> None:
        """Set active global color theme."""
        self.theme_name = theme_name

    # --- Candlestick & Price Action Charts ---

    def plot_candlestick(
        self,
        df: pd.DataFrame,
        chart_type: str = "ohlc",  # 'ohlc', 'heikin_ashi', 'renko'
        title: str = "Price Action Chart",
        brick_size: Optional[float] = None,
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render Candlestick chart (OHLC, Heikin Ashi, or Renko)."""
        start_t = time.perf_counter()
        chart_type_clean = chart_type.lower()

        if chart_type_clean == "heikin_ashi":
            fig = CandlestickRenderer.render_heikin_ashi(df, title=title, theme_name=self.theme_name, figsize=figsize)
        elif chart_type_clean == "renko":
            fig = CandlestickRenderer.render_renko(
                df, brick_size=brick_size, title=title, theme_name=self.theme_name, figsize=figsize
            )
        else:
            fig = CandlestickRenderer.render_ohlc(df, title=title, theme_name=self.theme_name, figsize=figsize)

        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("Candlestick", render_time)
        return fig

    # --- Backtest & Performance Charts ---

    def plot_equity_curve(
        self,
        equity_series: pd.Series,
        benchmark_series: Optional[pd.Series] = None,
        title: str = "Portfolio Equity Curve",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render Portfolio Equity Curve."""
        start_t = time.perf_counter()
        fig = EquityCurveRenderer.render_equity_curve(
            equity_series=equity_series,
            benchmark_series=benchmark_series,
            title=title,
            theme_name=self.theme_name,
            figsize=figsize,
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("EquityCurve", render_time)
        return fig

    def plot_trade_timeline(
        self,
        price_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        title: str = "Trade Execution Timeline",
        figsize: Tuple[float, float] = (12.0, 7.0),
    ) -> plt.Figure:
        """Render Trade Execution Timeline overlay."""
        start_t = time.perf_counter()
        fig = EquityCurveRenderer.render_trade_timeline(
            price_df=price_df, trades_df=trades_df, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("TradeTimeline", render_time)
        return fig

    def plot_drawdown(
        self,
        equity_series: pd.Series,
        title: str = "Underwater Drawdown Depth",
        figsize: Tuple[float, float] = (12.0, 5.0),
    ) -> plt.Figure:
        """Render Underwater Drawdown Depth chart."""
        start_t = time.perf_counter()
        fig = DrawdownRenderer.render_underwater_drawdown(
            equity_series=equity_series, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("Drawdown", render_time)
        return fig

    # --- Financial Statistics Charts ---

    def plot_returns_distribution(
        self,
        returns_series: pd.Series,
        title: str = "Daily Returns Distribution",
        figsize: Tuple[float, float] = (8.0, 5.0),
    ) -> plt.Figure:
        """Render returns distribution histogram with fitted density."""
        start_t = time.perf_counter()
        fig = StatisticsRenderer.render_returns_histogram(
            returns_series=returns_series, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("ReturnsDistribution", render_time)
        return fig

    def plot_rolling_sharpe(
        self,
        returns_series: pd.Series,
        window: int = 63,
        title: str = "Rolling Sharpe Ratio",
        figsize: Tuple[float, float] = (10.0, 4.5),
    ) -> plt.Figure:
        """Render rolling Sharpe ratio."""
        start_t = time.perf_counter()
        fig = StatisticsRenderer.render_rolling_sharpe(
            returns_series=returns_series, window=window, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("RollingSharpe", render_time)
        return fig

    # --- Correlation & Matrix Heatmaps ---

    def plot_correlation_matrix(
        self,
        returns_df: pd.DataFrame,
        method: str = "pearson",
        title: str = "Cross-Asset Correlation Matrix",
        figsize: Tuple[float, float] = (8.0, 7.0),
    ) -> plt.Figure:
        """Render correlation matrix heatmap."""
        start_t = time.perf_counter()
        fig = CorrelationRenderer.render_correlation_matrix(
            returns_df=returns_df, method=method, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("CorrelationMatrix", render_time)
        return fig

    # --- Machine Learning / Deep Learning Charts ---

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        title: str = "ML Feature Importance",
        figsize: Tuple[float, float] = (8.0, 5.0),
    ) -> plt.Figure:
        """Render Feature Importance bar chart."""
        start_t = time.perf_counter()
        fig = FeatureImportanceRenderer.render_feature_importance(
            importance_dict=importance_dict, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("FeatureImportance", render_time)
        return fig

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[float, float] = (6.0, 5.5),
    ) -> plt.Figure:
        """Render Confusion Matrix heatmap."""
        start_t = time.perf_counter()
        fig = FeatureImportanceRenderer.render_confusion_matrix(
            matrix=matrix, class_labels=class_labels, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("ConfusionMatrix", render_time)
        return fig

    def plot_learning_curve(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        title: str = "Learning Curve",
        figsize: Tuple[float, float] = (8.0, 4.5),
    ) -> plt.Figure:
        """Render Training vs Validation Loss learning curve."""
        start_t = time.perf_counter()
        fig = FeatureImportanceRenderer.render_learning_curve(
            train_loss=train_loss, val_loss=val_loss, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("LearningCurve", render_time)
        return fig

    # --- Monte Carlo & Walk Forward Charts ---

    def plot_monte_carlo(
        self,
        simulation_matrix: np.ndarray,
        title: str = "Monte Carlo Simulation Fan Chart",
        figsize: Tuple[float, float] = (12.0, 6.0),
    ) -> plt.Figure:
        """Render Monte Carlo simulation paths fan chart with confidence bands."""
        start_t = time.perf_counter()
        fig = MonteCarloRenderer.render_simulation_fan_chart(
            simulation_matrix=simulation_matrix, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("MonteCarlo", render_time)
        return fig

    def plot_walk_forward_windows(
        self,
        windows: List[Dict[str, Any]],
        title: str = "Walk Forward Window Timeline Split",
        figsize: Tuple[float, float] = (12.0, 5.0),
    ) -> plt.Figure:
        """Render Walk Forward window timeline split diagram."""
        start_t = time.perf_counter()
        fig = WalkForwardRenderer.render_window_timeline(
            windows=windows, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("WalkForward", render_time)
        return fig

    # --- Optimization Charts ---

    def plot_optimization_convergence(
        self,
        convergence_curve: List[Dict[str, Any]],
        title: str = "Optimization Fitness Convergence",
        figsize: Tuple[float, float] = (9.0, 5.0),
    ) -> plt.Figure:
        """Render Optimization fitness convergence curve."""
        start_t = time.perf_counter()
        fig = OptimizationRenderer.render_convergence(
            convergence_curve=convergence_curve, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("OptimizationConvergence", render_time)
        return fig

    def plot_pareto_front(
        self,
        obj1_vals: List[float],
        obj2_vals: List[float],
        obj1_name: str = "Sharpe Ratio",
        obj2_name: str = "Max Drawdown (%)",
        title: str = "Multi-Objective Pareto Front Analysis",
        figsize: Tuple[float, float] = (8.0, 6.0),
    ) -> plt.Figure:
        """Render Multi-Objective Pareto Front scatter plot."""
        start_t = time.perf_counter()
        fig = OptimizationRenderer.render_pareto_front(
            objective_1_values=obj1_vals,
            objective_2_values=obj2_vals,
            obj_1_name=obj1_name,
            obj_2_name=obj2_name,
            title=title,
            theme_name=self.theme_name,
            figsize=figsize,
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("ParetoFront", render_time)
        return fig

    # --- Portfolio Charts ---

    def plot_portfolio_allocation(
        self,
        allocations: Dict[str, float],
        title: str = "Portfolio Asset Allocation",
        figsize: Tuple[float, float] = (7.0, 6.0),
    ) -> plt.Figure:
        """Render Asset Allocation donut/pie chart."""
        start_t = time.perf_counter()
        fig = PortfolioRenderer.render_asset_allocation(
            allocations=allocations, title=title, theme_name=self.theme_name, figsize=figsize
        )
        render_time = (time.perf_counter() - start_t) * 1000.0
        logger.log_render("PortfolioAllocation", render_time)
        return fig

    # --- Export Helper Methods ---

    def export_figure(self, fig: plt.Figure, filepath: str, export_format: str = "png") -> str:
        """Export figure object to target format file.

        Args:
            fig: Matplotlib Figure instance.
            filepath: Destination file path.
            export_format: One of 'png', 'svg', 'pdf', 'html', 'markdown'.

        Returns:
            Absolute file path.
        """
        fmt = export_format.lower()
        if fmt == "svg":
            return VisualizationExporter.to_svg(fig, filepath)
        elif fmt == "pdf":
            return VisualizationExporter.to_pdf(fig, filepath)
        elif fmt == "html":
            return VisualizationExporter.to_html(fig, filepath)
        elif fmt == "markdown" or fmt == "md":
            return VisualizationExporter.to_markdown(fig, filepath)
        else:
            return VisualizationExporter.to_png(fig, filepath)

    def close_all(self) -> None:
        """Close all managed active figures and clean memory."""
        self.chart_manager.close_all()
