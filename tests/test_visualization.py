"""
QuantLab Master Visualization Engine Test Suite.

Validates all 20 components of the Visualization Engine:
Themes, VisualizationCache, ChartManager, CandlestickRenderer, EquityCurveRenderer, DrawdownRenderer,
TradeDistributionRenderer, CorrelationRenderer, HeatmapRenderer, FeatureImportanceRenderer,
MonteCarloRenderer, WalkForwardRenderer, OptimizationRenderer, PortfolioRenderer, StatisticsRenderer,
AnimationEngine, VisualizationExporter, and VisualizationEngine.
"""

import os
import shutil
import tempfile
import unittest
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
from visualization.montecarlo import MonteCarloRenderer
from visualization.optimization import OptimizationRenderer
from visualization.portfolio import PortfolioRenderer
from visualization.statistics import StatisticsRenderer
from visualization.themes import Theme, ThemeManager
from visualization.visualization_engine import VisualizationEngine
from visualization.walkforward import WalkForwardRenderer


class TestQuantLabVisualizationEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Visualization Engine."""

    def setUp(self) -> None:
        """Set up temporary output directory and synthetic datasets."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_vis_test_")

        # Synthetic OHLCV data
        n_bars = 50
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, size=n_bars))
        self.ohlc_df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": prices + 0.0008,
                "low": prices - 0.0008,
                "close": prices + 0.0002,
                "volume": 1000 + np.random.randint(0, 500, size=n_bars),
            },
            index=dates,
        )

        # Synthetic equity series
        self.equity = pd.Series(100000.0 + np.cumsum(np.random.normal(50, 200, size=n_bars)), index=dates)

    def tearDown(self) -> None:
        """Clean up temporary files and close all Matplotlib figures."""
        plt.close("all")
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_theme_manager(self) -> None:
        """Test ThemeManager built-in themes: Dark, Light, Institutional, Trading Desk, Research."""
        themes = ThemeManager.list_themes()
        self.assertTrue(len(themes) >= 5)

        dark_theme = ThemeManager.get_theme("dark")
        self.assertEqual(dark_theme.name, "Dark")

        fig, ax = plt.subplots()
        applied_theme = ThemeManager.apply(fig, "institutional")
        self.assertEqual(applied_theme.name, "Institutional")
        plt.close(fig)

    def test_visualization_cache(self) -> None:
        """Test VisualizationCache hashing, hit/miss telemetry, and clear operations."""
        cache = VisualizationCache()
        key = cache.generate_key("Candlestick", {"asset": "EURUSD"}, {"period": 20}, "dark")
        self.assertFalse(cache.contains(key))

        cache.put(key, "Candlestick", b"fake_bytes", 15.2)
        self.assertTrue(cache.contains(key))
        cached = cache.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.image_bytes, b"fake_bytes")

        stats = cache.statistics
        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["hits"], 1)

    def test_chart_manager(self) -> None:
        """Test ChartManager figure registration, subplots creation, and memory cleanup."""
        cm = ChartManager()
        fig, ax = cm.create_figure(title="Test Figure")
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        fig_sub, axes = cm.create_subplots(nrows=2, ncols=1)
        self.assertIsNotNone(fig_sub)
        self.assertEqual(len(axes), 2)

        cm.close_all()

    def test_candlestick_renderer(self) -> None:
        """Test CandlestickRenderer OHLC, Heikin Ashi, and Renko charts."""
        fig_ohlc = CandlestickRenderer.render_ohlc(self.ohlc_df)
        self.assertIsNotNone(fig_ohlc)
        plt.close(fig_ohlc)

        fig_ha = CandlestickRenderer.render_heikin_ashi(self.ohlc_df)
        self.assertIsNotNone(fig_ha)
        plt.close(fig_ha)

        fig_renko = CandlestickRenderer.render_renko(self.ohlc_df, brick_size=0.001)
        self.assertIsNotNone(fig_renko)
        plt.close(fig_renko)

    def test_equity_curve_and_drawdown_renderers(self) -> None:
        """Test EquityCurveRenderer and DrawdownRenderer graphs."""
        fig_eq = EquityCurveRenderer.render_equity_curve(self.equity)
        self.assertIsNotNone(fig_eq)
        plt.close(fig_eq)

        trades_df = pd.DataFrame(
            [
                {"entry_idx": 5, "exit_idx": 15, "side": "BUY", "entry_price": 1.10, "exit_price": 1.12, "pnl": 200.0},
                {"entry_idx": 20, "exit_idx": 30, "side": "SELL", "entry_price": 1.12, "exit_price": 1.11, "pnl": 100.0},
            ]
        )
        fig_timeline = EquityCurveRenderer.render_trade_timeline(self.ohlc_df, trades_df)
        self.assertIsNotNone(fig_timeline)
        plt.close(fig_timeline)

        fig_dd = DrawdownRenderer.render_underwater_drawdown(self.equity)
        self.assertIsNotNone(fig_dd)
        plt.close(fig_dd)

    def test_correlation_and_heatmap_renderers(self) -> None:
        """Test CorrelationRenderer matrix heatmaps and HeatmapRenderer parameter grids."""
        returns_df = pd.DataFrame(
            {
                "EURUSD": np.random.normal(0.001, 0.01, 50),
                "GBPUSD": np.random.normal(0.001, 0.012, 50),
                "USDJPY": np.random.normal(-0.0005, 0.008, 50),
            }
        )
        fig_corr = CorrelationRenderer.render_correlation_matrix(returns_df, method="pearson")
        self.assertIsNotNone(fig_corr)
        plt.close(fig_corr)

        grid_df = pd.DataFrame(
            {
                "period": [10, 10, 20, 20],
                "threshold": [1.0, 2.0, 1.0, 2.0],
                "fitness_score": [50.0, 75.0, 60.0, 90.0],
            }
        )
        fig_heat = HeatmapRenderer.render_parameter_heatmap(grid_df, "period", "threshold", "fitness_score")
        self.assertIsNotNone(fig_heat)
        plt.close(fig_heat)

    def test_ml_and_monte_carlo_renderers(self) -> None:
        """Test FeatureImportanceRenderer and MonteCarloRenderer graphics."""
        imp_dict = {"feature_rsi": 0.45, "feature_macd": 0.35, "feature_vol": 0.20}
        fig_imp = FeatureImportanceRenderer.render_feature_importance(imp_dict)
        self.assertIsNotNone(fig_imp)
        plt.close(fig_imp)

        cm_matrix = np.array([[45, 5], [10, 40]])
        fig_cm = FeatureImportanceRenderer.render_confusion_matrix(cm_matrix)
        self.assertIsNotNone(fig_cm)
        plt.close(fig_cm)

        # Monte Carlo simulation paths
        sim_matrix = 100000.0 + np.cumsum(np.random.normal(50, 300, size=(20, 50)), axis=1)
        fig_mc = MonteCarloRenderer.render_simulation_fan_chart(sim_matrix)
        self.assertIsNotNone(fig_mc)
        plt.close(fig_mc)

    def test_walkforward_and_optimization_renderers(self) -> None:
        """Test WalkForwardRenderer window timelines and OptimizationRenderer Pareto front plots."""
        windows = [
            {"train_start": 0, "train_end": 100, "val_start": 100, "val_end": 130},
            {"train_start": 30, "train_end": 130, "val_start": 130, "val_end": 160},
        ]
        fig_wf = WalkForwardRenderer.render_window_timeline(windows)
        self.assertIsNotNone(fig_wf)
        plt.close(fig_wf)

        fig_pareto = OptimizationRenderer.render_pareto_front([1.2, 1.8, 2.1], [25.0, 18.0, 12.0])
        self.assertIsNotNone(fig_pareto)
        plt.close(fig_pareto)

    def test_portfolio_and_statistics_renderers(self) -> None:
        """Test PortfolioRenderer asset allocation pie charts and StatisticsRenderer return histograms."""
        alloc = {"EURUSD": 40.0, "GBPUSD": 30.0, "USDJPY": 30.0}
        fig_port = PortfolioRenderer.render_asset_allocation(alloc)
        self.assertIsNotNone(fig_port)
        plt.close(fig_port)

        returns = self.equity.pct_change()
        fig_stat = StatisticsRenderer.render_returns_histogram(returns)
        self.assertIsNotNone(fig_stat)
        plt.close(fig_stat)

    def test_animation_engine(self) -> None:
        """Test AnimationEngine dynamic replay animation creation."""
        anim = AnimationEngine.create_candlestick_replay_animation(self.ohlc_df.head(10), interval_ms=50)
        self.assertIsNotNone(anim)

    def test_visualization_exporter(self) -> None:
        """Test VisualizationExporter PNG, SVG, PDF, HTML, Interactive HTML, Markdown exports."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        png_p = VisualizationExporter.to_png(fig, os.path.join(self.temp_dir, "chart.png"))
        self.assertTrue(os.path.exists(png_p))

        svg_p = VisualizationExporter.to_svg(fig, os.path.join(self.temp_dir, "chart.svg"))
        self.assertTrue(os.path.exists(svg_p))

        pdf_p = VisualizationExporter.to_pdf(fig, os.path.join(self.temp_dir, "chart.pdf"))
        self.assertTrue(os.path.exists(pdf_p))

        html_p = VisualizationExporter.to_html(fig, os.path.join(self.temp_dir, "chart.html"))
        self.assertTrue(os.path.exists(html_p))

        i_html_p = VisualizationExporter.to_interactive_html(fig, os.path.join(self.temp_dir, "chart_i.html"))
        self.assertTrue(os.path.exists(i_html_p))

        md_p = VisualizationExporter.to_markdown(fig, os.path.join(self.temp_dir, "chart.md"))
        self.assertTrue(os.path.exists(md_p))

        plt.close(fig)

    def test_master_visualization_engine(self) -> None:
        """Test master VisualizationEngine rendering calls, theme switches, and export helpers."""
        engine = VisualizationEngine(theme_name="institutional")

        fig_candle = engine.plot_candlestick(self.ohlc_df, chart_type="ohlc")
        self.assertIsNotNone(fig_candle)

        fig_eq = engine.plot_equity_curve(self.equity)
        self.assertIsNotNone(fig_eq)

        exp_path = os.path.join(self.temp_dir, "master_chart.png")
        saved_path = engine.export_figure(fig_candle, exp_path, export_format="png")
        self.assertTrue(os.path.exists(saved_path))

        engine.close_all()


if __name__ == "__main__":
    unittest.main()
