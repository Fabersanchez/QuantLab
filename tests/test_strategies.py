"""
QuantLab Strategy Framework Unit Tests.

Verifies functionality of BaseStrategy, StrategyMetadata, StrategyRegistry,
StrategyValidator, StrategyBuilder, StrategyComposition, StrategyExporter,
StrategyOptimizer, StrategyPipeline, and StrategyEngine using standard library unittest.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from strategies import (
    BaseStrategy,
    StrategyMetadata,
    StrategyRegistry,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
    StrategyValidator,
    StrategyBuilder,
    StrategyExporter,
    OptimizerFactory,
    StrategyEngine,
    PriceThresholdCondition,
    IndicatorCrossCondition,
    TrendFilter,
    MultiIndicatorConfirmation,
    FixedFractionRisk,
)


class SampleTrendStrategy(BaseStrategy):
    """Sample strategy for testing framework functionality."""

    @classmethod
    def metadata(cls) -> StrategyMetadata:
        return StrategyMetadata(
            name="SampleTrendStrategy",
            category="TrendFollowing",
            description="Sample Trend Strategy testing framework execution.",
            indicators_required=["sma_10"],
        )

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        prep = data.copy()
        if "sma_10" not in [c.lower() for c in prep.columns]:
            c_col = [c for c in prep.columns if c.lower() == "close"][0]
            prep["sma_10"] = prep[c_col].rolling(10).mean().fillna(prep[c_col])
        return prep

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        c_col = [c for c in data.columns if c.lower() == "close"][0]
        c = data[c_col]
        sma = data["sma_10"]
        signal = np.where(c > sma, 1, np.where(c < sma, -1, 0))
        return pd.DataFrame({"signal": signal}, index=data.index)


class TestQuantLabStrategyFramework(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        timestamps = pd.date_range("2026-01-01 09:30", periods=50, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(50))
        high_prices = close_prices + np.abs(np.random.randn(50))
        low_prices = close_prices - np.abs(np.random.randn(50))
        open_prices = low_prices + (high_prices - low_prices) * np.random.rand(50)
        volume = 1000.0 + np.random.rand(50) * 500.0

        self.sample_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            }
        )
        self.engine = StrategyEngine()
        self.engine.register_strategy(SampleTrendStrategy, overwrite=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_registry(self) -> None:
        registry = StrategyRegistry()
        registry.register(SampleTrendStrategy)
        self.assertTrue(registry.has("SampleTrendStrategy"))
        self.assertEqual(registry.get("SampleTrendStrategy"), SampleTrendStrategy)
        self.assertTrue(registry.is_enabled("SampleTrendStrategy"))

        registry.disable("SampleTrendStrategy")
        self.assertFalse(registry.is_enabled("SampleTrendStrategy"))

        with self.assertRaises(StrategyAlreadyRegisteredError):
            registry.register(SampleTrendStrategy)

    def test_strategy_execution(self) -> None:
        strat = SampleTrendStrategy()
        output_df = strat.execute(self.sample_df)
        self.assertIn("signal", output_df.columns)
        self.assertIn("sma_10", output_df.columns)
        self.assertEqual(len(output_df), 50)

    def test_strategy_builder_and_composition(self) -> None:
        builder = self.engine.new_builder("ComposedTrend")
        builder.add_entry_condition(PriceThresholdCondition("close", "open", ">"))
        builder.set_risk_rule(FixedFractionRisk(0.01))

        composed_strat = builder.build()
        output_df = composed_strat.execute(self.sample_df)
        self.assertIn("signal", output_df.columns)

    def test_strategy_validator(self) -> None:
        validator = StrategyValidator()
        strat = SampleTrendStrategy()
        prepared = strat.prepare(self.sample_df)
        report = validator.validate(strat, prepared)
        self.assertTrue(report.is_valid)

    def test_strategy_exporter(self) -> None:
        strat = SampleTrendStrategy()
        json_path = os.path.join(self.temp_dir, "strategy.json")
        json_str = StrategyExporter.to_json(strat, json_path)
        self.assertIn("SampleTrendStrategy", json_str)
        self.assertTrue(os.path.exists(json_path))

        sentinel_spec = StrategyExporter.to_sentinel_spec(strat)
        self.assertIn("sentinel_version", sentinel_spec)

    def test_optimizer_factory(self) -> None:
        grid_opt = OptimizerFactory.get_optimizer("grid")
        self.assertIsNotNone(grid_opt)
        res = grid_opt.optimize(SampleTrendStrategy, {}, self.sample_df)
        self.assertIsNotNone(res)

    def test_strategy_engine_execute(self) -> None:
        output_df = self.engine.execute_strategy("SampleTrendStrategy", self.sample_df)
        self.assertIn("signal", output_df.columns)


if __name__ == "__main__":
    unittest.main()
