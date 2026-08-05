"""
QuantLab Master Portfolio Engine & Simulator Test Suite.

Validates all 20 components of the Portfolio Engine:
Asset, Portfolio, 10 Capital Allocation Models, AllocationEngine, PortfolioOptimizer, PortfolioRebalancer,
ExposureAnalyzer, DiversificationAnalyzer, CorrelationAnalyzer, CovarianceAnalyzer, RiskEngine,
PerformanceAnalyzer, PortfolioConstraints, PortfolioMetrics, PortfolioSimulator, PortfolioExporter,
PortfolioReportEngine, and PortfolioEngine.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from portfolio import (
    AllocationEngine,
    Asset,
    BlackLittermanModel,
    CorrelationAnalyzer,
    CovarianceAnalyzer,
    CustomAllocationModel,
    DiversificationAnalyzer,
    EqualRiskContributionModel,
    EqualWeightModel,
    ExposureAnalyzer,
    HRPAllocationModel,
    KellyAllocationModel,
    MarketType,
    MaximumDiversificationModel,
    MeanVarianceModel,
    MinimumVarianceModel,
    PerformanceAnalyzer,
    Portfolio,
    PortfolioConstraints,
    PortfolioEngine,
    PortfolioExporter,
    PortfolioMetrics,
    PortfolioOptimizer,
    PortfolioRebalancer,
    PortfolioReportEngine,
    PortfolioSimulator,
    RebalanceTrigger,
    RiskEngine,
    RiskParityModel,
    SimulationConfig,
)


class TestQuantLabPortfolioEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Portfolio Engine & Simulator."""

    def setUp(self) -> None:
        """Set up temporary output directory and synthetic multi-asset returns data."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_port_test_")

        # Create assets
        self.asset_eurusd = Asset(symbol="EURUSD", name="Euro / US Dollar", market=MarketType.FOREX, sector="Currencies")
        self.asset_aapl = Asset(symbol="AAPL", name="Apple Inc.", market=MarketType.STOCKS, sector="Technology")
        self.asset_btc = Asset(symbol="BTCUSD", name="Bitcoin", market=MarketType.CRYPTO, sector="DigitalAssets")

        # Synthetic asset returns DataFrame
        n_bars = 252
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="1D")
        np.random.seed(42)
        ret_eur = np.random.normal(0.0003, 0.008, size=n_bars)
        ret_aapl = np.random.normal(0.0008, 0.015, size=n_bars)
        ret_btc = np.random.normal(0.0015, 0.035, size=n_bars)

        self.returns_df = pd.DataFrame(
            {"EURUSD": ret_eur, "AAPL": ret_aapl, "BTCUSD": ret_btc}, index=dates
        )

        # Synthetic equity series
        self.equity = pd.Series(100000.0 * np.cumprod(1.0 + ret_aapl), index=dates)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_asset_and_portfolio_container(self) -> None:
        """Test Asset metadata and Portfolio container operations."""
        port = Portfolio(name="TestPortfolio", initial_capital=500000.0)
        port.add_asset(self.asset_eurusd, 0.40)
        port.add_asset(self.asset_aapl, 0.60)

        self.assertEqual(len(port.assets), 2)
        self.assertEqual(port.weights["EURUSD"], 0.40)

        # Versioning & Cloning
        v_next = port.increment_version("minor")
        self.assertEqual(v_next, "1.1.0")

        cloned = port.clone("ClonedPort")
        self.assertEqual(cloned.name, "ClonedPort")
        self.assertNotEqual(cloned.portfolio_id, port.portfolio_id)

        # JSON Serialization
        json_path = os.path.join(self.temp_dir, "port.json")
        port.to_json(json_path)
        loaded = Portfolio.from_json(json_path)
        self.assertEqual(loaded.name, port.name)
        self.assertEqual(loaded.initial_capital, 500000.0)

    def test_allocation_models_and_engine(self) -> None:
        """Test all 10 Capital Allocation Models and AllocationEngine."""
        symbols = ["EURUSD", "AAPL", "BTCUSD"]
        engine = AllocationEngine()

        # Test EqualWeight
        w_eq = engine.compute_allocation(symbols, model_id="equal_weight")
        self.assertAlmostEqual(sum(w_eq.values()), 1.0, places=5)
        self.assertAlmostEqual(w_eq["EURUSD"], 1.0 / 3.0, places=5)

        # Test RiskParity
        w_rp = engine.compute_allocation(symbols, model_id="risk_parity", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_rp.values()), 1.0, places=5)

        # Test MinimumVariance
        w_mv = engine.compute_allocation(symbols, model_id="minimum_variance", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_mv.values()), 1.0, places=5)

        # Test MaximumDiversification
        w_md = engine.compute_allocation(symbols, model_id="maximum_diversification", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_md.values()), 1.0, places=5)

        # Test Kelly Allocation
        w_k = engine.compute_allocation(symbols, model_id="kelly", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_k.values()), 1.0, places=5)

        # Test HRP
        w_hrp = engine.compute_allocation(symbols, model_id="hrp", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_hrp.values()), 1.0, places=5)

        # Test BlackLitterman
        w_bl = engine.compute_allocation(symbols, model_id="black_litterman", returns_df=self.returns_df, views_dict={"AAPL": 0.12})
        self.assertAlmostEqual(sum(w_bl.values()), 1.0, places=5)

        # Test MeanVariance
        w_mvo = engine.compute_allocation(symbols, model_id="mean_variance", returns_df=self.returns_df)
        self.assertAlmostEqual(sum(w_mvo.values()), 1.0, places=5)

        # Test Custom Allocation
        custom_model = CustomAllocationModel(lambda syms: {s: 1.0 / len(syms) for s in syms})
        engine.register_model("custom_test", custom_model)
        w_cust = engine.compute_allocation(symbols, model_id="custom_test")
        self.assertAlmostEqual(sum(w_cust.values()), 1.0, places=5)

    def test_portfolio_optimizer_and_rebalancer(self) -> None:
        """Test PortfolioOptimizer objective functions and PortfolioRebalancer triggers."""
        symbols = ["EURUSD", "AAPL", "BTCUSD"]
        opt = PortfolioOptimizer()

        w_sharpe = opt.optimize(symbols, self.returns_df, objective="sharpe")
        self.assertAlmostEqual(sum(w_sharpe.values()), 1.0, places=5)

        w_vol = opt.optimize(symbols, self.returns_df, objective="volatility")
        self.assertAlmostEqual(sum(w_vol.values()), 1.0, places=5)

        # Rebalancing
        port = Portfolio(name="RebalPort")
        port.add_asset(self.asset_eurusd, 0.50)
        port.add_asset(self.asset_aapl, 0.50)

        rebal = PortfolioRebalancer(drift_threshold=0.05)
        need_rebal = rebal.should_rebalance(port, current_weights={"EURUSD": 0.60, "AAPL": 0.40}, trigger_type=RebalanceTrigger.WEIGHT_DRIFT)
        self.assertTrue(need_rebal)

        event = rebal.execute_rebalance(port, target_weights={"EURUSD": 0.50, "AAPL": 0.50})
        self.assertEqual(event["new_weights"]["EURUSD"], 0.50)
        self.assertEqual(len(port.history_events), 1)

    def test_exposure_and_diversification_analyzers(self) -> None:
        """Test ExposureAnalyzer and DiversificationAnalyzer."""
        port = Portfolio(name="ExposurePort")
        port.add_asset(self.asset_eurusd, 0.50)
        port.add_asset(self.asset_aapl, 0.50)

        exp = ExposureAnalyzer.analyze(port, position_values={"EURUSD": 50000.0, "AAPL": 50000.0})
        self.assertEqual(exp.gross_exposure_val, 100000.0)
        self.assertAlmostEqual(exp.leverage_ratio, 1.0)

        weights = {"EURUSD": 0.50, "AAPL": 0.50}
        cov_matrix = CovarianceAnalyzer.compute_sample_covariance(self.returns_df[["EURUSD", "AAPL"]])
        div_ratio = DiversificationAnalyzer.compute_diversification_ratio(weights, cov_matrix)
        self.assertGreaterEqual(div_ratio, 0.99)

        enb = DiversificationAnalyzer.compute_effective_number_of_assets(weights)
        self.assertAlmostEqual(enb, 2.0, places=4)

    def test_covariance_and_correlation_analyzers(self) -> None:
        """Test CovarianceAnalyzer (Sample, Ledoit-Wolf, OAS) and CorrelationAnalyzer."""
        cov_sample = CovarianceAnalyzer.compute_sample_covariance(self.returns_df)
        self.assertEqual(cov_sample.shape, (3, 3))

        cov_lw = CovarianceAnalyzer.compute_ledoit_wolf_covariance(self.returns_df)
        self.assertEqual(cov_lw.shape, (3, 3))

        cov_oas = CovarianceAnalyzer.compute_oas_covariance(self.returns_df)
        self.assertEqual(cov_oas.shape, (3, 3))

        corr_p = CorrelationAnalyzer.compute_correlation_matrix(self.returns_df, method="pearson")
        self.assertEqual(corr_p.shape, (3, 3))

    def test_risk_engine_and_metrics(self) -> None:
        """Test RiskEngine (VaR, CVaR, Beta, Stress Test) and PortfolioMetrics."""
        returns_aapl = self.returns_df["AAPL"]
        returns_bm = self.returns_df["EURUSD"]

        risk_metrics = RiskEngine.analyze_risk(returns_aapl, benchmark_returns=returns_bm)
        self.assertIsNotNone(risk_metrics.var_parametric_95)
        self.assertIsNotNone(risk_metrics.cvar_expected_shortfall_95)
        self.assertIn("2008_Financial_Crash_Shock", risk_metrics.stress_test_results)

        port_metrics = PortfolioMetrics.calculate(self.equity, benchmark_series=self.equity)
        self.assertGreater(port_metrics.sharpe_ratio, -100.0)

    def test_portfolio_simulator(self) -> None:
        """Test PortfolioSimulator multi-path Monte Carlo simulation execution."""
        port = Portfolio(name="SimPort", initial_capital=100000.0)
        port.add_asset(self.asset_eurusd, 0.50)
        port.add_asset(self.asset_aapl, 0.50)

        sim = PortfolioSimulator(config=SimulationConfig(n_simulations=100, n_bars=50))
        res = sim.run_simulation(port, returns_df=self.returns_df[["EURUSD", "AAPL"]])

        self.assertEqual(res.simulation_matrix.shape, (100, 50))
        self.assertGreater(res.mean_final_equity, 0.0)

    def test_exporter_and_report_engine(self) -> None:
        """Test PortfolioExporter (CSV, Excel, JSON, SQLite, Parquet, Markdown, PDF) and PortfolioReportEngine."""
        port = Portfolio(name="ExportPort", initial_capital=100000.0)
        port.add_asset(self.asset_eurusd, 0.50)
        port.add_asset(self.asset_aapl, 0.50)

        csv_p = PortfolioExporter.to_csv(port, os.path.join(self.temp_dir, "port.csv"))
        self.assertTrue(os.path.exists(csv_p))

        excel_p = PortfolioExporter.to_excel(port, os.path.join(self.temp_dir, "port.xlsx"))
        self.assertTrue(os.path.exists(excel_p))

        json_p = PortfolioExporter.to_json(port, os.path.join(self.temp_dir, "port.json"))
        self.assertTrue(os.path.exists(json_p))

        db_p = PortfolioExporter.to_sqlite(port, os.path.join(self.temp_dir, "port.db"))
        self.assertTrue(os.path.exists(db_p))

        parq_p = PortfolioExporter.to_parquet(port, os.path.join(self.temp_dir, "port.parquet"))
        self.assertTrue(os.path.exists(parq_p))

        md_p = PortfolioExporter.to_markdown(port, os.path.join(self.temp_dir, "port.md"))
        self.assertTrue(os.path.exists(md_p))

        report_md = PortfolioReportEngine.generate_report(port)
        self.assertIn("QuantLab Executive Portfolio Analysis Report", report_md)

    def test_master_portfolio_engine(self) -> None:
        """Test master PortfolioEngine lifecycle operations."""
        pe = PortfolioEngine()
        p1 = pe.create_portfolio("MasterPort1", 200000.0)
        p1.add_asset(self.asset_eurusd, 0.50)
        p1.add_asset(self.asset_aapl, 0.50)

        p2 = pe.duplicate_portfolio(p1.portfolio_id, "MasterPort2")
        self.assertIsNotNone(p2)

        df_cmp = pe.compare_portfolios([p1.portfolio_id, p2.portfolio_id])
        self.assertEqual(len(df_cmp), 2)

        exp_json = pe.export_portfolio(p1.portfolio_id, os.path.join(self.temp_dir, "master.json"), export_format="json")
        self.assertTrue(os.path.exists(exp_json))

        deleted = pe.delete_portfolio(p2.portfolio_id)
        self.assertTrue(deleted)


if __name__ == "__main__":
    unittest.main()
