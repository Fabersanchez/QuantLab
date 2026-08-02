"""
QuantLab Monte Carlo & Robustness Engine Unit Tests.

Verifies functionality of bootstrap samplers, permutation algorithms, noise injectors,
stress testing, sensitivity analysis, SimulationRunner (100 to 1,000+ iterations),
distribution metrics, probability calculations (PoP, PoR), confidence intervals (90%, 95%, 99%),
Institutional Robustness Score, Fan Chart visualizer, report generator, and master MonteCarloEngine.
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

from monte_carlo import (
    # Bootstrap & Permutation
    RandomReplacementBootstrap,
    BlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    BootstrapSamplerFactory,
    TradeOrderPermutation,
    ReturnsPermutation,
    # Noise Injectors & Stress
    SpreadNoiseInjector,
    SlippageNoiseInjector,
    CompositeNoiseInjector,
    StressScenarioType,
    StressTestRunner,
    SensitivityAnalyzer,
    # Runner & Scenario Generator
    ScenarioType,
    ScenarioGenerator,
    SimulationRunner,
    # Metrics & Probabilities
    DistributionMetricsCalculator,
    ProbabilityAnalyzer,
    ConfidenceIntervalCalculator,
    InstitutionalRobustnessScore,
    # Visuals & Reports
    MonteCarloVisualizer,
    MonteCarloReportGenerator,
    # Engine
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
)


class SampleStrategy(BaseStrategy):
    """Sample strategy for Monte Carlo testing."""

    @classmethod
    def metadata(cls) -> StrategyMetadata:
        return StrategyMetadata(
            name="SampleStrategy",
            category="Test",
            description="Sample strategy for Monte Carlo testing.",
        )

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = np.zeros(len(data), dtype=int)
        if len(data) > 5:
            signals[5] = 1
        if len(data) > 15:
            signals[15] = -1
        return pd.DataFrame({"signal": signals}, index=data.index)


class TestMonteCarloRobustnessEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        timestamps = pd.date_range("2026-01-01 09:30", periods=50, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(50) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(50) * 0.2) + 0.1
        low_prices = close_prices - np.abs(np.random.randn(50) * 0.2) - 0.1
        open_prices = low_prices + (high_prices - low_prices) * 0.5
        volume = np.random.randint(1000, 5000, size=50).astype(float)

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
        self.strategy = SampleStrategy()
        self.sample_trades = [150.0, -50.0, 200.0, -100.0, 300.0, -80.0, 120.0, 50.0, -150.0, 220.0]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_bootstrap_samplers(self) -> None:
        series = pd.Series(self.sample_trades)

        # Standard Replacement
        iid = BootstrapSamplerFactory.create("random")
        sample_iid = iid.sample(series)
        self.assertEqual(len(sample_iid), len(series))

        # Block Bootstrap
        blk = BootstrapSamplerFactory.create("block", block_size=3)
        sample_blk = blk.sample(series)
        self.assertEqual(len(sample_blk), len(series))

        # Moving Block Bootstrap
        m_blk = BootstrapSamplerFactory.create("moving_block", block_size=3)
        sample_mblk = m_blk.sample(series)
        self.assertEqual(len(sample_mblk), len(series))

        # Stationary Bootstrap
        stat = BootstrapSamplerFactory.create("stationary", avg_block_size=4.0)
        sample_stat = stat.sample(series)
        self.assertEqual(len(sample_stat), len(series))

    def test_permutations_and_noise_injectors(self) -> None:
        perm_trades = TradeOrderPermutation.permute(self.sample_trades)
        self.assertEqual(len(perm_trades), len(self.sample_trades))
        self.assertEqual(sum(perm_trades), sum(self.sample_trades))

        spread_inj = SpreadNoiseInjector(noise_std_pips=1.0)
        p_spread = spread_inj.perturb(0.0001)
        self.assertGreaterEqual(p_spread, 0.0)

        comp_inj = CompositeNoiseInjector()
        bar_dict = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "spread": 0.0001}
        p_bar = comp_inj.apply_bar_noise(bar_dict)
        self.assertIn("high", p_bar)

    def test_stress_testing(self) -> None:
        stress_results = StressTestRunner.run_all_stress_tests(self.strategy, self.dataset)
        self.assertEqual(len(stress_results), len(StressScenarioType))
        self.assertTrue(hasattr(stress_results[0], "survived"))

    def test_sensitivity_analysis(self) -> None:
        sens_output = SensitivityAnalyzer.analyze_execution_cost_sensitivity(
            self.strategy, self.dataset, spread_pips_range=[0.0, 1.0, 2.0]
        )
        self.assertIn("spread", sens_output)
        self.assertEqual(len(sens_output["spread"]), 3)

        elasticity = SensitivityAnalyzer.calculate_elasticity_score(sens_output["spread"])
        self.assertGreaterEqual(elasticity, 0.0)

    def test_simulation_runner(self) -> None:
        runner = SimulationRunner(initial_capital=100000.0, ruin_threshold_pct=50.0)
        # Run 500 simulation iterations
        sim_results = runner.run_simulations_from_trades(
            trades=self.sample_trades, n_iterations=500, scenario_type=ScenarioType.TRADE_PERMUTATION
        )
        self.assertEqual(len(sim_results), 500)
        self.assertEqual(sim_results[0].iteration_id, 0)
        self.assertIsInstance(sim_results[0].final_equity, float)

    def test_metrics_probabilities_and_confidence(self) -> None:
        runner = SimulationRunner(initial_capital=100000.0)
        sim_results = runner.run_simulations_from_trades(self.sample_trades, n_iterations=100)

        # Distribution Metrics
        dist_m = DistributionMetricsCalculator.calculate_distribution_metrics(sim_results)
        self.assertIn("expected_return", dist_m)
        self.assertIn("mean_max_drawdown_pct", dist_m)

        # Probabilities
        probs = ProbabilityAnalyzer.calculate_all_probabilities(sim_results)
        self.assertIn("probability_of_profit_pct", probs)
        self.assertIn("probability_of_ruin_pct", probs)

        # Confidence Intervals
        ci = ConfidenceIntervalCalculator.calculate_confidence_intervals(sim_results)
        self.assertIn("net_profit", ci)
        self.assertIn("95%", ci["net_profit"])

        # Robustness Score
        r_score = InstitutionalRobustnessScore.calculate_score(sim_results)
        self.assertIn("institutional_robustness_score", r_score)
        self.assertGreaterEqual(r_score["institutional_robustness_score"], 0.0)

    def test_visualization_and_report_generator(self) -> None:
        runner = SimulationRunner(initial_capital=100000.0)
        sim_results = runner.run_simulations_from_trades(self.sample_trades, n_iterations=100)

        # Visualizer
        viz = MonteCarloVisualizer(sim_results)
        df_fan = viz.compute_equity_fan_bands()
        self.assertIn("p50", df_fan.columns)

        fan_svg = viz.generate_equity_fan_chart_svg()
        self.assertIn("<svg", fan_svg)

        # Report Generator
        engine = MonteCarloEngine()
        engine.load_dataset_and_strategy(self.dataset, self.strategy)
        res = engine.start_monte_carlo()

        reporter = MonteCarloReportGenerator(res)
        out_dir = os.path.join(self.temp_dir, "mc_reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["equity_fan_bands_csv"]))

    def test_monte_carlo_engine(self) -> None:
        config = MonteCarloConfig(
            iterations=200,
            scenario_type=ScenarioType.TRADE_PERMUTATION,
            run_stress_tests=False,
            run_sensitivity=False,
        )
        engine = MonteCarloEngine(config)
        engine.load_dataset_and_strategy(self.dataset, self.strategy)

        res = engine.start_monte_carlo()
        self.assertIsInstance(res, MonteCarloResult)
        self.assertEqual(res.total_iterations, 200)
        self.assertGreater(res.execution_time_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
