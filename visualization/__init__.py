"""
QuantLab Master Visualization Engine Package.

Provides centralized, high-performance financial graphics, interactive charts,
statistical distributions, ML/DL curves, optimization surfaces, Monte Carlo bands,
Walk Forward timelines, portfolio allocations, animations, and multi-format exports.
"""

from visualization.animation import AnimationEngine
from visualization.candlestick import CandlestickRenderer
from visualization.cache import CachedChart, VisualizationCache
from visualization.chart_manager import ChartManager
from visualization.correlation import CorrelationRenderer
from visualization.drawdown import DrawdownRenderer
from visualization.equity_curve import EquityCurveRenderer
from visualization.exporter import VisualizationExporter
from visualization.feature_importance import FeatureImportanceRenderer
from visualization.heatmap import HeatmapRenderer
from visualization.logger import VisualizationLogger, get_visualization_logger
from visualization.montecarlo import MonteCarloRenderer
from visualization.optimization import OptimizationRenderer
from visualization.portfolio import PortfolioRenderer
from visualization.statistics import StatisticsRenderer
from visualization.themes import Theme, ThemeManager
from visualization.visualization_engine import VisualizationEngine
from visualization.walkforward import WalkForwardRenderer

__all__ = [
    "VisualizationEngine",
    "ChartManager",
    "ThemeManager",
    "Theme",
    "VisualizationCache",
    "CachedChart",
    "VisualizationExporter",
    "AnimationEngine",
    "VisualizationLogger",
    "get_visualization_logger",
    "CandlestickRenderer",
    "EquityCurveRenderer",
    "DrawdownRenderer",
    "TradeDistributionRenderer",
    "CorrelationRenderer",
    "HeatmapRenderer",
    "FeatureImportanceRenderer",
    "MonteCarloRenderer",
    "WalkForwardRenderer",
    "OptimizationRenderer",
    "PortfolioRenderer",
    "StatisticsRenderer",
]
