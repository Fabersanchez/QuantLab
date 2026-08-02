"""
QuantLab Walk Forward Optimization Engine Unit Tests.

Verifies functionality of rolling, expanding, anchored, sliding, and custom window generators,
optimizer adapters (Grid, Random, Optuna), ValidationRunner, robustness metrics, efficiency analytics,
visualizers, report generators, and master WalkForwardEngine using standard library unittest.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy
from strategies.strategy_metadata import StrategyMetadata

from walk_forward import (
    WindowSplit,
    WindowGeneratorFactory,
    RollingWindowGenerator,
    SlidingWindowGenerator,
    ExpandingWindowGenerator,
    AnchoredWindowGenerator,
    CustomWindowGenerator,
    BaseOptimizerAdapter,
    GridSearchOptimizerAdapter,
    RandomSearchOptimizerAdapter,
    OptunaOptimizerAdapter,
    OptimizerAdapterFactory,
    ValidationRunner,
    WindowStatisticsCalculator,
    RobustnessMetricsCalculator,
    EfficiencyAnalyzer,
    WalkForwardVisualizer,
    WalkForwardReportGenerator,
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
)


class DummyParamStrategy(BaseStrategy):
    """Parametric strategy for testing Walk Forward optimization."""

    @classmethod
    def metadata(cls) -> StrategyMetadata:
        return StrategyMetadata(
            name="DummyParamStrategy",
            category="Test",
            description="Dummy strategy with tunable period parameter.",
        )

    @classmethod
    def default_parameters(cls) -> dict:
        return {"period": 5}

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        period = int(self.params.get("period", 5))
        c_col = "close" if "close" in data.columns else data.columns[0]
        close = data[c_col]
        sma = close.rolling(max(1, period)).mean().fillna(close)
        signal = np.where(close > sma, 1, np.where(close < sma, -1, 0))
        return pd.DataFrame({"signal": signal}, index=data.index)


class TestWalkForwardOptimizationEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        timestamps = pd.date_range("2026-01-01 09:30", periods=200, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(200) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(200) * 0.2) + 0.1
        low_prices = close_prices - np.abs(np.random.randn(200) * 0.2) - 0.1
        open_prices = low_prices + (high_prices - low_prices) * 0.5
        volume = np.random.randint(1000, 5000, size=200).astype(float)

        self.df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            }
        )

        self.dataset = MarketDataset(
            data=self.df, asset="EURUSD", timeframe="1m", broker="GenericTest"
        )
        self.strategy = DummyParamStrategy()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_window_generators(self) -> None:
        # Rolling Window
        rolling = WindowGeneratorFactory.create("rolling", train_bars=50, val_bars=20, step_bars=20)
        r_splits = rolling.generate_windows(self.df)
        self.assertGreater(len(r_splits), 0)
        self.assertEqual(r_splits[0].train_bars, 50)
        self.assertEqual(r_splits[0].val_bars, 20)

        # Sliding Window
        sliding = SlidingWindowGenerator(train_bars=50, val_bars=20, overlap_bars=5)
        s_splits = sliding.generate_windows(self.df)
        self.assertGreater(len(s_splits), 0)

        # Expanding Window
        expanding = WindowGeneratorFactory.create("expanding", initial_train_bars=50, val_bars=20, step_bars=20)
        e_splits = expanding.generate_windows(self.df)
        self.assertGreater(len(e_splits), 0)
        self.assertEqual(e_splits[1].train_start_index, 0)
        self.assertGreater(e_splits[1].train_bars, e_splits[0].train_bars)

        # Anchored Window
        anchored = WindowGeneratorFactory.create("anchored", anchor_index=0, initial_train_bars=50, val_bars=20)
        a_splits = anchored.generate_windows(self.df)
        self.assertGreater(len(a_splits), 0)

        # Custom Window
        custom = CustomWindowGenerator(explicit_bounds=[(0, 49, 50, 69), (20, 69, 70, 89)])
        c_splits = custom.generate_windows(self.df)
        self.assertEqual(len(c_splits), 2)

    def test_optimizer_adapters(self) -> None:
        param_grid = {"period": [3, 5, 10]}

        # Grid Search
        grid = OptimizerAdapterFactory.create("grid")
        best_p, score, metrics = grid.optimize(
            DummyParamStrategy, param_grid, self.df.iloc[:50], asset_symbol="EURUSD"
        )
        self.assertIn("period", best_p)
        self.assertIn(best_p["period"], [3, 5, 10])

        # Random Search
        rand_opt = OptimizerAdapterFactory.create("random", max_evals=2)
        r_best_p, r_score, r_metrics = rand_opt.optimize(
            DummyParamStrategy, param_grid, self.df.iloc[:50], asset_symbol="EURUSD"
        )
        self.assertIn("period", r_best_p)

        # Optuna Adapter
        optuna_opt = OptimizerAdapterFactory.create("optuna", n_trials=2)
        o_best_p, o_score, o_metrics = optuna_opt.optimize(
            DummyParamStrategy, param_grid, self.df.iloc[:50], asset_symbol="EURUSD"
        )
        self.assertIn("period", o_best_p)

    def test_validation_runner(self) -> None:
        rolling = RollingWindowGenerator(train_bars=50, val_bars=20, step_bars=30)
        windows = rolling.generate_windows(self.df)

        runner = ValidationRunner(optimizer_adapter=GridSearchOptimizerAdapter())
        results = runner.run_validation_sequence(
            strategy_cls=DummyParamStrategy,
            param_grid={"period": [3, 5]},
            dataset=self.dataset,
            windows=windows,
        )

        self.assertEqual(len(results), len(windows))
        self.assertIn("period", results[0].best_params)
        self.assertIsNotNone(results[0].oos_metrics)

    def test_robustness_and_efficiency_metrics(self) -> None:
        rolling = RollingWindowGenerator(train_bars=50, val_bars=20, step_bars=30)
        windows = rolling.generate_windows(self.df)

        runner = ValidationRunner()
        results = runner.run_validation_sequence(
            strategy_cls=DummyParamStrategy,
            param_grid={"period": [3, 5]},
            dataset=self.dataset,
            windows=windows,
        )

        # Statistics summary
        summary_df = WindowStatisticsCalculator.compute_summary_table(results)
        self.assertEqual(len(summary_df), len(results))

        agg_stats = WindowStatisticsCalculator.compute_aggregate_statistics(results)
        self.assertEqual(agg_stats["window_count"], len(results))

        # Robustness metrics
        rob = RobustnessMetricsCalculator.calculate_all(results, len(self.df))
        self.assertIn("walk_forward_efficiency_pct", rob)
        self.assertIn("robustness_index", rob)
        self.assertIn("overfitting_score_pct", rob)

        # Efficiency metrics
        stitched_eq = pd.DataFrame({"equity": [100000.0, 101000.0, 102000.0]})
        eff = EfficiencyAnalyzer.analyze_efficiency(results, stitched_eq)
        self.assertIn("return_wfe_pct", eff)
        self.assertIn("rolling_cagr", eff)

    def test_visualization_and_reports(self) -> None:
        engine = WalkForwardEngine()
        engine.config.train_bars = 50
        engine.config.val_bars = 20
        engine.config.step_bars = 30
        engine.set_param_grid({"period": [3, 5]})

        engine.load_dataset(self.dataset)
        engine.load_strategy(self.strategy)
        res = engine.start_walkforward()

        # Test Visualizer
        viz = WalkForwardVisualizer(res.window_results)
        svg_eq = viz.generate_equity_comparison_svg(res.concatenated_oos_equity)
        self.assertIn("<svg", svg_eq)

        svg_cmp = viz.generate_window_comparison_svg()
        self.assertIn("<svg", svg_cmp)

        # Test Report Generator
        reporter = WalkForwardReportGenerator(res)
        out_dir = os.path.join(self.temp_dir, "wfa_reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["window_summary_csv"]))

    def test_walk_forward_engine(self) -> None:
        config = WalkForwardConfig(
            train_bars=50,
            val_bars=20,
            step_bars=30,
            window_type="rolling",
            optimizer_type="grid",
            param_grid={"period": [3, 5]},
        )

        engine = WalkForwardEngine(config)
        engine.load_dataset(self.dataset)
        engine.load_strategy(self.strategy)

        res = engine.start_walkforward()
        self.assertIsInstance(res, WalkForwardResult)
        self.assertEqual(res.strategy_name, "DummyParamStrategy")
        self.assertGreater(res.execution_time_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
