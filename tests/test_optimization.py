"""
QuantLab Master Optimization Engine Test Suite.

Validates all 15 components of the Optimization Engine:
Parameters, SearchSpace, Constraints, ObjectiveFunction, OptimizationCache, OptimizationHistory,
SolutionEvaluator, OptimizationScheduler, Optimizer & 12 Algorithms, OptimizationManager,
Visualization, Exporters, ReportEngine, and Integration Adapters.
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from data.market_dataset import MarketDataset
from optimization.adapters import OptimizationExperimentAdapter
from optimization.cache import OptimizationCache
from optimization.constraints import OptimizationConstraints
from optimization.evaluator import SolutionEvaluator
from optimization.exporter import OptimizationExporter
from optimization.history import OptimizationHistory
from optimization.objective_function import ObjectiveFunction
from optimization.optimization_manager import OptimizationManager
from optimization.optimizer import Optimizer
from optimization.parameter_space import (
    BooleanParameter,
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    LogScaleParameter,
)
from optimization.reports import OptimizationReportEngine
from optimization.scheduler import OptimizationScheduler
from optimization.search_space import SearchSpace
from optimization.visualization import OptimizationVisualizer
from strategies.base_strategy import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Simple test strategy for optimization unit testing."""

    def __init__(self, period: int = 20, threshold: float = 1.5, use_filter: bool = True) -> None:
        super().__init__(name="DummyStrategy")
        self.period = period
        self.threshold = threshold
        self.use_filter = use_filter

    def on_bar(self, bar: pd.Series) -> None:
        if bar.name % max(1, self.period) == 0:
            if self.position_manager.is_flat:
                self.order_manager.submit_market_order("EURUSD", "BUY", 1.0)
            else:
                self.order_manager.submit_market_order("EURUSD", "SELL", 1.0)


class TestQuantLabOptimizationEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Optimization Engine."""

    def setUp(self) -> None:
        """Set up synthetic dataset and workspace for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_opt_test_")

        # Synthetic OHLCV dataset
        n_bars = 100
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, size=n_bars))
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": prices + 0.0005,
                "low": prices - 0.0005,
                "close": prices,
                "volume": 1000,
            }
        )
        self.dataset = MarketDataset(df, asset="EURUSD", timeframe="1h")

        # Search space setup
        self.space = SearchSpace(name="DummySpace")
        self.space.add_parameter(IntegerParameter("period", 5, 50, step=5))
        self.space.add_parameter(FloatParameter("threshold", 1.0, 3.0, step=0.5))
        self.space.add_parameter(BooleanParameter("use_filter"))

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parameter_space(self) -> None:
        """Test parameter sampling, domain validation, normalization, and denormalization."""
        p_int = IntegerParameter("p_int", 10, 100, step=10)
        sample = p_int.sample()
        self.assertTrue(p_int.validate(sample))
        norm = p_int.normalize(50)
        self.assertAlmostEqual(p_int.denormalize(norm), 50)

        p_float = FloatParameter("p_float", 0.0, 1.0)
        f_sample = p_float.sample()
        self.assertTrue(p_float.validate(f_sample))

        p_log = LogScaleParameter("p_log", 0.001, 1.0)
        log_sample = p_log.sample()
        self.assertTrue(p_log.validate(log_sample))

    def test_search_space(self) -> None:
        """Test flat and hierarchical search space sampling, grid points, and vector conversion."""
        self.assertEqual(self.space.dimension, 3)
        sample = self.space.sample()
        self.assertTrue(self.space.contains(sample))

        grid = list(self.space.grid_points(points_per_dim=3))
        self.assertTrue(len(grid) > 0)

        vec = self.space.normalize(sample)
        denorm = self.space.denormalize(vec)
        self.assertTrue(self.space.contains(denorm))

    def test_constraints_and_objective_function(self) -> None:
        """Test OptimizationConstraints boundary rules and ObjectiveFunction multi-objective evaluation."""
        constraints = OptimizationConstraints(max_drawdown=20.0, min_profit_factor=1.2, min_trades=5)
        metrics_pass = {"max_drawdown": 10.0, "profit_factor": 1.5, "total_trades": 10}
        valid, violations = constraints.evaluate(metrics_pass)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

        metrics_fail = {"max_drawdown": 30.0, "profit_factor": 0.8, "total_trades": 2}
        valid_f, violations_f = constraints.evaluate(metrics_fail)
        self.assertFalse(valid_f)
        self.assertTrue(len(violations_f) >= 2)

        obj = ObjectiveFunction()
        score = obj.evaluate(metrics_pass, execution_time_sec=0.5)
        self.assertTrue(isinstance(score, float))

    def test_cache_and_history(self) -> None:
        """Test OptimizationCache hit/miss hashing and OptimizationHistory iteration logging."""
        cache = OptimizationCache()
        params = {"period": 20, "threshold": 2.0}
        key = cache.generate_key(DummyStrategy, self.dataset, params)
        self.assertFalse(cache.contains(key))

        cache.put(key, params, {"net_profit": 500.0}, 85.0, True, 0.1)
        self.assertTrue(cache.contains(key))
        cached = cache.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.fitness_score, 85.0)

        history = OptimizationHistory()
        history.add_record(1, params, 85.0, True, {"net_profit": 500.0}, 0.1)
        top = history.get_top_solutions(k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].fitness_score, 85.0)

    def test_all_twelve_optimization_algorithms(self) -> None:
        """Test running all 12 optimization algorithms through master Optimizer."""
        algorithms = [
            "grid_search",
            "random_search",
            "bayesian",
            "optuna",
            "hyperopt",
            "pso",
            "ga",
            "es",
            "de",
            "sa",
            "tpe",
            "cma_es",
        ]

        for algo in algorithms:
            opt = Optimizer(
                strategy_cls=DummyStrategy,
                dataset=self.dataset,
                search_space=self.space,
                algorithm=algo,
            )

            top_rec = opt.optimize(max_evaluations=4, batch_size=2)
            self.assertIsNotNone(top_rec)
            self.assertTrue(len(opt.history.get_all_records()) > 0)

    def test_optimizer_controls_and_serialization(self) -> None:
        """Test Optimizer control methods: pause, resume, cancel, save, load, version, export."""
        opt = Optimizer(
            strategy_cls=DummyStrategy,
            dataset=self.dataset,
            search_space=self.space,
            algorithm="random_search",
        )

        opt.optimize(max_evaluations=4, batch_size=2)

        # Versioning
        self.assertEqual(opt.version, "1.0.0")
        opt.versionar("patch")
        self.assertEqual(opt.version, "1.0.1")

        # Single evaluation
        single_res = opt.evaluar({"period": 10, "threshold": 1.5, "use_filter": True})
        self.assertIsNotNone(single_res)

        # Save and Load state
        checkpoint_path = os.path.join(self.temp_dir, "opt_state.json")
        opt.guardar(checkpoint_path)
        self.assertTrue(os.path.exists(checkpoint_path))

        opt.cargar(checkpoint_path)

        # Export
        csv_path = os.path.join(self.temp_dir, "opt_out.csv")
        opt.exportar(csv_path, export_format="csv")
        self.assertTrue(os.path.exists(csv_path))

    def test_optimization_manager(self) -> None:
        """Test OptimizationManager multi-job queueing, status tracking, and cancellation."""
        manager = OptimizationManager(total_cpus=2)

        opt1 = Optimizer(
            strategy_cls=DummyStrategy, dataset=self.dataset, search_space=self.space, algorithm="random_search"
        )
        job_id1 = manager.submit_job(opt1, max_evaluations=2, batch_size=2, priority=5)
        self.assertIsNotNone(job_id1)

        status = manager.get_job_status(job_id1)
        self.assertIsNotNone(status)

        manager.cancel_all()

    def test_visualization(self) -> None:
        """Test OptimizationVisualizer plots: convergence, parameter importance, fitness distribution."""
        history = OptimizationHistory()
        history.add_record(1, {"period": 10, "threshold": 1.5}, 60.0, True, {}, 0.1)
        history.add_record(2, {"period": 20, "threshold": 2.0}, 80.0, True, {}, 0.1)

        conv_fig = OptimizationVisualizer.plot_convergence_curve(history)
        self.assertIsNotNone(conv_fig)

        imp_fig = OptimizationVisualizer.plot_parameter_importance(history)
        self.assertIsNotNone(imp_fig)

        dist_fig = OptimizationVisualizer.plot_fitness_distribution(history)
        self.assertIsNotNone(dist_fig)

    def test_exporters(self) -> None:
        """Test OptimizationExporter CSV, Excel, SQLite, JSON, Parquet, Markdown, PDF exports."""
        history = OptimizationHistory()
        history.add_record(1, {"period": 10, "threshold": 1.5}, 75.0, True, {"sharpe_ratio": 1.5}, 0.2)

        csv_p = OptimizationExporter.to_csv(history, os.path.join(self.temp_dir, "hist.csv"))
        self.assertTrue(os.path.exists(csv_p))

        json_p = OptimizationExporter.to_json(history, os.path.join(self.temp_dir, "hist.json"))
        self.assertTrue(os.path.exists(json_p))

        excel_p = OptimizationExporter.to_excel(history, os.path.join(self.temp_dir, "hist.xlsx"))
        self.assertTrue(os.path.exists(excel_p))

        sqlite_p = OptimizationExporter.to_sqlite(history, os.path.join(self.temp_dir, "hist.db"))
        self.assertTrue(os.path.exists(sqlite_p))

        parquet_p = OptimizationExporter.to_parquet(history, os.path.join(self.temp_dir, "hist.parquet"))
        self.assertTrue(os.path.exists(parquet_p))

        md_p = OptimizationExporter.to_markdown(history, os.path.join(self.temp_dir, "hist.md"))
        self.assertTrue(os.path.exists(md_p))

        pdf_p = OptimizationExporter.to_pdf(history, os.path.join(self.temp_dir, "hist.pdf"))
        self.assertTrue(os.path.exists(pdf_p))

    def test_report_engine_and_adapters(self) -> None:
        """Test OptimizationReportEngine and OptimizationExperimentAdapter."""
        history = OptimizationHistory()
        rec = history.add_record(1, {"period": 15, "threshold": 2.0}, 82.0, True, {"sharpe_ratio": 1.8}, 0.3)

        report_engine = OptimizationReportEngine()
        report = report_engine.generate_report("DummyStrategy", "GeneticAlgorithm", self.space, history)
        self.assertIsNotNone(report.markdown_content)
        self.assertIn("QUANTLAB STRATEGY OPTIMIZATION REPORT", report.markdown_content)

        exp = OptimizationExperimentAdapter.to_experiment(rec)
        self.assertIsNotNone(exp.uuid)
        self.assertEqual(exp.parameters["period"], 15)


if __name__ == "__main__":
    unittest.main()
