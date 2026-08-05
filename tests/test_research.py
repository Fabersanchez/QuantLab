"""
QuantLab Master Research Engine Test Suite.

Validates all 14 components of the Research Engine laboratory:
Experiment entity serialization (JSON, YAML, SQLite), ExperimentRegistry persistence & SQL queries,
ExperimentManager lifecycle & thread safety, MetricsCalculator & 30+ decoupled metrics, Comparator ranking,
Validator multi-criteria rules, Scorer 0-100 ratings, MetadataExtractor, ReproducibilityManager,
Exporter formats (CSV, Excel, JSON, Markdown, SQLite, Parquet, PDF), ReportEngine, and Integration Adapters.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from research.adapters import (
    BacktestExperimentAdapter,
    MLExperimentAdapter,
    MonteCarloExperimentAdapter,
    WalkForwardExperimentAdapter,
)
from research.comparator import Comparator, ComparisonResult
from research.experiment import Experiment, ExperimentStatus
from research.experiment_manager import ExperimentManager
from research.experiment_registry import ExperimentRegistry
from research.exporter import ExperimentExporter
from research.logger import get_research_logger
from research.metadata import MetadataExtractor, SystemMetadata
from research.metrics import MetricsCalculator, ProfitFactorMetric, SharpeRatioMetric
from research.reports import ReportEngine, ResearchReport
from research.reproducibility import ReproducibilityManager
from research.scorer import ScoreResult, ScoreWeights, Scorer
from research.validator import ValidationResult, ValidationRule, Validator


class TestQuantLabResearchEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Research Engine."""

    def setUp(self) -> None:
        """Set up temporary working workspace for research exports and DB testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_research_test_")
        self.db_path = os.path.join(self.temp_dir, "test_research.db")
        self.registry = ExperimentRegistry(db_path=self.db_path)
        self.manager = ExperimentManager(registry=self.registry, max_workers=2)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_experiment_creation_and_serialization(self) -> None:
        """Test Experiment entity creation, hashing, cloning, versioning, and JSON/YAML/SQLite serialization."""
        exp = Experiment(
            name="Alpha_MeanReversion",
            description="Mean reversion on EURUSD 1h",
            author="QuantResearchTeam",
            asset="EURUSD",
            timeframe="1h",
            parameters={"period": 20, "std_dev": 2.0},
            random_seed=123,
            results={"sharpe_ratio": 1.85, "max_drawdown": 12.4, "profit_factor": 1.72, "net_profit": 15400.0},
        )

        self.assertIsNotNone(exp.uuid)
        self.assertTrue(len(exp.hash) > 0)
        self.assertTrue(len(exp.checksum) > 0)

        # Versioning test
        self.assertEqual(exp.version, "1.0.0")
        exp.increment_version("patch")
        self.assertEqual(exp.version, "1.0.1")

        # Cloning test
        cloned = exp.clone(new_name="Alpha_MeanReversion_V2", param_overrides={"period": 30})
        self.assertNotEqual(exp.uuid, cloned.uuid)
        self.assertEqual(cloned.name, "Alpha_MeanReversion_V2")
        self.assertEqual(cloned.parameters["period"], 30)

        # JSON Serialization
        json_file = os.path.join(self.temp_dir, "exp.json")
        exp.to_json(filepath=json_file)
        exp_from_json = Experiment.from_json(json_file)
        self.assertEqual(exp.uuid, exp_from_json.uuid)
        self.assertEqual(exp.name, exp_from_json.name)

        # YAML Serialization
        yaml_file = os.path.join(self.temp_dir, "exp.yaml")
        exp.to_yaml(filepath=yaml_file)
        exp_from_yaml = Experiment.from_yaml(yaml_file)
        self.assertEqual(exp.uuid, exp_from_yaml.uuid)

        # SQLite Serialization
        sqlite_file = os.path.join(self.temp_dir, "exp_direct.db")
        exp.to_sqlite(sqlite_file)
        exp_from_sqlite = Experiment.from_sqlite(sqlite_file, exp.uuid)
        self.assertEqual(exp.uuid, exp_from_sqlite.uuid)

    def test_experiment_registry_and_persistence(self) -> None:
        """Test ExperimentRegistry persistence, SQL queries, history logging, and resource usage tracking."""
        exp = Experiment(
            name="TrendFollower",
            author="DevTeam",
            asset="GBPUSD",
            results={"sharpe_ratio": 1.45, "net_profit": 8500.0},
            resource_metrics={"ram_peak_mb": 128.5, "cpu_usage_pct": 14.2},
        )

        self.registry.register(exp, log_message="Registration test.")
        fetched = self.registry.get(exp.uuid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "TrendFollower")

        # Query testing
        queried = self.registry.query(name="TrendFollower", asset="GBPUSD")
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried[0].uuid, exp.uuid)

        # History test
        exp.status = ExperimentStatus.COMPLETED
        self.registry.update(exp, log_message="Completed backtest stage.")
        history = self.registry.get_history(exp.uuid)
        self.assertTrue(len(history) >= 2)

        # Resource consumption query test
        res_usage = self.registry.get_resource_consumption(exp.uuid)
        self.assertEqual(res_usage["ram_peak_mb"], 128.5)

    def test_experiment_manager_lifecycle_and_concurrency(self) -> None:
        """Test ExperimentManager creation, status transitions, search, list, clone, and concurrent execution."""
        exp1 = self.manager.create_experiment(name="Exp_1", asset="EURUSD")
        exp2 = self.manager.create_experiment(name="Exp_2", asset="USDJPY")

        self.assertEqual(exp1.status, ExperimentStatus.INITIALIZED)

        # Pause and resume
        self.manager.pause_experiment(exp1.uuid)
        self.assertEqual(self.manager.get_experiment(exp1.uuid).status, ExperimentStatus.PAUSED)
        self.manager.resume_experiment(exp1.uuid)
        self.assertEqual(self.manager.get_experiment(exp1.uuid).status, ExperimentStatus.RUNNING)

        # Search and listing
        results = self.manager.search_experiments(asset="EURUSD")
        self.assertEqual(len(results), 1)

        listed = self.manager.list_experiments()
        self.assertTrue(len(listed) >= 2)

        # Concurrent execution test
        def dummy_task(e: Experiment):
            return {"net_profit": 5000.0, "sharpe_ratio": 1.2}

        completed = self.manager.run_concurrently(dummy_task, [exp1, exp2])
        self.assertEqual(len(completed), 2)
        for c in completed:
            self.assertEqual(c.status, ExperimentStatus.COMPLETED)
            self.assertEqual(c.results["net_profit"], 5000.0)

    def test_metrics_calculator_and_decoupled_metrics(self) -> None:
        """Test MetricsCalculator and 30+ decoupled independent metrics."""
        # Create synthetic trades DataFrame
        trades_data = {
            "pnl": [100.0, -50.0, 200.0, -30.0, 150.0, -40.0, 300.0],
            "pnl_pct": [0.01, -0.005, 0.02, -0.003, 0.015, -0.004, 0.03],
            "duration": [10, 5, 12, 4, 8, 6, 15],
            "commission": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "slippage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
        trades_df = pd.DataFrame(trades_data)

        # Synthetic equity series
        equity_series = pd.Series([100000.0, 100100.0, 100050.0, 100250.0, 100220.0, 100370.0, 100330.0, 100630.0])

        calc = MetricsCalculator()
        computed = calc.compute_all(
            trades_df=trades_df,
            equity_series=equity_series,
            initial_capital=100000.0,
            total_bars=500,
        )

        self.assertIn("profit_factor", computed)
        self.assertIn("sharpe_ratio", computed)
        self.assertIn("max_drawdown", computed)
        self.assertIn("win_rate", computed)
        self.assertIn("sqn", computed)
        self.assertIn("kelly_criterion", computed)

        # Profit Factor verify
        expected_gross_profit = 100.0 + 200.0 + 150.0 + 300.0  # 750
        expected_gross_loss = 50.0 + 30.0 + 40.0  # 120
        self.assertAlmostEqual(computed["profit_factor"], 750.0 / 120.0)

        # Win Rate verify
        self.assertAlmostEqual(computed["win_rate"], (4.0 / 7.0) * 100.0)

    def test_comparator_and_benchmarking(self) -> None:
        """Test Comparator multi-experiment benchmarking, composite scoring, and rankings."""
        exp_a = Experiment(
            name="Strategy_A",
            results={"sharpe_ratio": 2.1, "max_drawdown": 10.0, "net_profit": 20000.0, "profit_factor": 2.2},
            execution_time=1.5,
        )
        exp_b = Experiment(
            name="Strategy_B",
            results={"sharpe_ratio": 1.2, "max_drawdown": 25.0, "net_profit": 8000.0, "profit_factor": 1.3},
            execution_time=3.0,
        )

        comparator = Comparator()
        res: ComparisonResult = comparator.compare([exp_a, exp_b])

        self.assertEqual(res.experiments_count, 2)
        self.assertEqual(res.winner_uuid, exp_a.uuid)
        self.assertEqual(res.rankings[0]["name"], "Strategy_A")
        self.assertEqual(res.rankings[0]["rank"], 1)

    def test_validator_and_anti_winrate_policy(self) -> None:
        """Test Validator multi-criteria evaluation and anti-WinRate enforcement."""
        # Experiment with 90% WinRate but terrible risk profile (huge drawdown, negative expectancy)
        bad_exp = Experiment(
            name="HighWinRate_BadRisk",
            results={
                "win_rate": 90.0,
                "profit_factor": 0.8,
                "sharpe_ratio": -0.2,
                "max_drawdown": 45.0,
                "expectancy": -15.0,
                "recovery_factor": 0.2,
                "calmar_ratio": -0.1,
            },
        )

        validator = Validator()
        val_res: ValidationResult = validator.validate(bad_exp)

        self.assertEqual(val_res.status, "REJECTED")
        self.assertEqual(bad_exp.status, ExperimentStatus.REJECTED)
        self.assertTrue(len(val_res.failed_rules) > 0)

        # Good experiment passing all default rules
        good_exp = Experiment(
            name="Institutional_Strategy",
            results={
                "win_rate": 55.0,
                "profit_factor": 1.8,
                "sharpe_ratio": 1.5,
                "max_drawdown": 12.0,
                "expectancy": 45.0,
                "recovery_factor": 2.5,
                "calmar_ratio": 1.2,
            },
        )
        val_res_good = validator.validate(good_exp)
        self.assertEqual(val_res_good.status, "PASSED")

    def test_scorer_ratings_and_grades(self) -> None:
        """Test Scorer 0-100 institutional ratings and letter grades."""
        exp = Experiment(
            name="TopStrategy",
            results={
                "profit_factor": 2.2,
                "sharpe_ratio": 2.1,
                "sortino_ratio": 2.8,
                "calmar_ratio": 2.0,
                "recovery_factor": 3.5,
                "max_drawdown": 8.0,
                "expectancy": 120.0,
                "win_rate": 62.0,
                "risk_reward": 1.8,
                "walk_forward_efficiency": 0.85,
                "monte_carlo_ruin_prob": 0.01,
                "sqn": 3.2,
            },
        )

        scorer = Scorer()
        score_res: ScoreResult = scorer.score(exp)

        self.assertTrue(0.0 <= score_res.overall_score <= 100.0)
        self.assertTrue(0.0 <= score_res.institutional_score <= 100.0)
        self.assertTrue(0.0 <= score_res.robustness_score <= 100.0)
        self.assertIn(score_res.grade, ["S", "A", "B", "C", "D", "F"])
        self.assertTrue(score_res.overall_score >= 70.0)  # Should score grade B or higher

    def test_metadata_and_reproducibility(self) -> None:
        """Test MetadataExtractor system collection and ReproducibilityManager deterministic contract."""
        meta: SystemMetadata = MetadataExtractor.collect(broker="InteractiveBrokers", random_seed=999)
        self.assertEqual(meta.broker, "InteractiveBrokers")
        self.assertEqual(meta.random_seed, 999)
        self.assertTrue(meta.ram_total_gb > 0)
        self.assertTrue(meta.cpu_cores_logical > 0)

        # Reproducibility test
        config = {"lr": 0.01, "batch_size": 32}
        ds_info = {"asset": "EURUSD", "rows": 1000}
        ctx1 = ReproducibilityManager.capture_context(seed=42, config=config, dataset_repr=ds_info)
        ctx2 = ReproducibilityManager.capture_context(seed=42, config=config, dataset_repr=ds_info)

        verification = ReproducibilityManager.verify_reproducibility(ctx1, ctx2)
        self.assertTrue(verification["is_reproducible"])
        self.assertTrue(verification["seed_match"])
        self.assertTrue(verification["config_match"])
        self.assertTrue(verification["dataset_match"])

    def test_exporter_multi_format(self) -> None:
        """Test ExperimentExporter across CSV, Excel, JSON, Markdown, SQLite, Parquet, and PDF."""
        exp = Experiment(
            name="ExportStrategy",
            results={"sharpe_ratio": 1.6, "net_profit": 12000.0, "profit_factor": 1.8},
            parameters={"ma_fast": 10, "ma_slow": 50},
        )

        json_p = ExperimentExporter.to_json(exp, os.path.join(self.temp_dir, "exp_export.json"))
        self.assertTrue(os.path.exists(json_p))

        csv_p = ExperimentExporter.to_csv(exp, os.path.join(self.temp_dir, "exp_export.csv"))
        self.assertTrue(os.path.exists(csv_p))

        excel_p = ExperimentExporter.to_excel(exp, os.path.join(self.temp_dir, "exp_export.xlsx"))
        self.assertTrue(os.path.exists(excel_p))

        md_p = ExperimentExporter.to_markdown(exp, os.path.join(self.temp_dir, "exp_export.md"))
        self.assertTrue(os.path.exists(md_p))

        sqlite_p = ExperimentExporter.to_sqlite(exp, os.path.join(self.temp_dir, "exp_export.db"))
        self.assertTrue(os.path.exists(sqlite_p))

        parquet_p = ExperimentExporter.to_parquet(exp, os.path.join(self.temp_dir, "exp_export.parquet"))
        self.assertTrue(os.path.exists(parquet_p))

        pdf_p = ExperimentExporter.to_pdf(exp, os.path.join(self.temp_dir, "exp_export.pdf"))
        self.assertTrue(os.path.exists(pdf_p))

    def test_report_engine(self) -> None:
        """Test ReportEngine scientific research report generation."""
        exp = Experiment(
            name="Institutional_Report_Strategy",
            results={
                "sharpe_ratio": 1.7,
                "profit_factor": 1.9,
                "max_drawdown": 11.5,
                "net_profit": 18500.0,
                "win_rate": 58.0,
                "expectancy": 42.0,
                "recovery_factor": 2.8,
                "calmar_ratio": 1.4,
            },
        )

        engine = ReportEngine()
        report: ResearchReport = engine.generate_report(exp)

        self.assertIsNotNone(report.markdown_content)
        self.assertIn("QUANTLAB SCIENTIFIC RESEARCH REPORT", report.markdown_content)
        self.assertIn(report.recommendation, ["APPROVED_FOR_PRODUCTION", "NEEDS_REVISION", "REJECTED"])

        md_save_path = os.path.join(self.temp_dir, "research_report.md")
        saved_path = report.save_markdown(md_save_path)
        self.assertTrue(os.path.exists(saved_path))

    def test_integration_adapters(self) -> None:
        """Test integration adapters converting engine results into institutional Experiments."""
        # Mock BacktestResult object
        class MockBacktestResult:
            strategy_name = "SMA_Cross"
            asset_symbol = "EURUSD"
            timeframe = "1h"
            metrics = {"sharpe_ratio": 1.6, "profit_factor": 1.75}
            statistics = {"net_profit": 14000.0}
            execution_time_seconds = 2.45

        bt_res = MockBacktestResult()
        bt_exp = BacktestExperimentAdapter.to_experiment(bt_res)
        self.assertIsInstance(bt_exp, Experiment)
        self.assertEqual(bt_exp.results["sharpe_ratio"], 1.6)
        self.assertEqual(bt_exp.results["net_profit"], 14000.0)

        # Mock WalkForwardResult object
        class MockWFResult:
            strategy_name = "WF_Opt"
            asset_symbol = "GBPUSD"
            robustness_metrics = {"robustness_index": 0.82}
            efficiency_metrics = {"walk_forward_efficiency": 0.78}
            statistics_summary = {"mean_oos_return": 0.08}
            execution_time_seconds = 12.3

        wf_res = MockWFResult()
        wf_exp = WalkForwardExperimentAdapter.to_experiment(wf_res)
        self.assertEqual(wf_exp.results["walk_forward_efficiency"], 0.78)

        # Mock MonteCarloResult object
        class MockMCResult:
            strategy_name = "MC_Sim"
            asset_symbol = "USDJPY"
            distribution_metrics = {"mean_profit": 15000.0}
            probability_metrics = {"ruin_probability": 0.005}
            robustness_score = {"institutional_score": 88.0}
            execution_time_seconds = 5.6

        mc_res = MockMCResult()
        mc_exp = MonteCarloExperimentAdapter.to_experiment(mc_res)
        self.assertEqual(mc_exp.results["ruin_probability"], 0.005)


if __name__ == "__main__":
    unittest.main()
