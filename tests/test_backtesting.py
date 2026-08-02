"""
QuantLab Institutional Backtesting Engine Unit Tests.

Verifies complete functionality of commission models, slippage models, spread models,
latency models, OrderManager, PositionManager, PortfolioManager, ExecutionSimulator,
EquityCurve, TradeLog, PerformanceMetrics, BacktestStatistics, ReportGenerator,
BacktestReplay, BacktestEngine, and BacktestRunner using standard library unittest.
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

from backtesting import (
    # Models
    FixedCommissionModel,
    PercentageCommissionModel,
    PerLotCommissionModel,
    BrokerSpecificCommissionModel,
    CustomCommissionModel,
    CommissionModelFactory,
    FixedSlippageModel,
    DynamicSlippageModel,
    VolatilityBasedSlippageModel,
    LiquidityBasedSlippageModel,
    RandomSlippageModel,
    SlippageModelFactory,
    FixedSpreadModel,
    VariableSpreadModel,
    HistoricalSpreadModel,
    BrokerSpreadModel,
    SpreadModelFactory,
    ExecutionDelayModel,
    NetworkDelayModel,
    ExchangeDelayModel,
    BrokerDelayModel,
    CompositeLatencyModel,
    LatencyModelFactory,
    # Managers & Data Structures
    Order,
    OrderManager,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    Position,
    PositionManager,
    PositionSide,
    PortfolioManager,
    PortfolioState,
    # Execution & Analytics
    ExecutionSimulator,
    EquityCurve,
    TradeLog,
    PerformanceMetrics,
    BacktestStatistics,
    ReportGenerator,
    BacktestReplay,
    ReplaySnapshot,
    # Engine & Runner
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestRunner,
)


class DummyTestStrategy(BaseStrategy):
    """Simple strategy for backtest testing."""

    @classmethod
    def metadata(cls) -> StrategyMetadata:
        return StrategyMetadata(
            name="DummyTestStrategy",
            category="Test",
            description="Dummy strategy generating alternating buy/sell signals.",
        )

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = np.zeros(len(data), dtype=int)
        # Buy on bar 5, Sell on bar 25
        if len(data) > 5:
            signals[5] = 1
        if len(data) > 25:
            signals[25] = -1
        return pd.DataFrame({"signal": signals}, index=data.index)


class TestInstitutionalBacktestingEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        timestamps = pd.date_range("2026-01-01 09:30", periods=60, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(60) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(60) * 0.2) + 0.1
        low_prices = close_prices - np.abs(np.random.randn(60) * 0.2) - 0.1
        open_prices = low_prices + (high_prices - low_prices) * 0.5
        volume = np.random.randint(1000, 5000, size=60).astype(float)

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
        self.strategy = DummyTestStrategy()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_commission_models(self) -> None:
        fixed = CommissionModelFactory.create("fixed", fee_per_order=2.5)
        self.assertEqual(fixed.calculate(100, 50.0), 2.5)

        pct = CommissionModelFactory.create("percentage", percentage=0.001)
        self.assertAlmostEqual(pct.calculate(10, 100.0), 1.0)

        lot = CommissionModelFactory.create("per_lot", cost_per_lot=7.0, lot_size=100000.0)
        self.assertAlmostEqual(lot.calculate(100000.0, 1.10), 7.0)

        broker = CommissionModelFactory.create("broker", broker_profile="forex_ecn_raw")
        self.assertAlmostEqual(broker.calculate(100000.0, 1.10), 6.0)

        custom = CommissionModelFactory.create("custom", fn=lambda q, p, s, n: 5.0)
        self.assertEqual(custom.calculate(10, 10), 5.0)

    def test_slippage_models(self) -> None:
        fixed = SlippageModelFactory.create("fixed", pips=2.0, point_value=0.0001)
        exec_buy = fixed.calculate_execution_price(1.1000, 1.0, "BUY")
        self.assertAlmostEqual(exec_buy, 1.1002)

        dyn = SlippageModelFactory.create("dynamic", base_percentage=0.001)
        exec_p = dyn.calculate_execution_price(100.0, 100.0, "BUY", {"volume": 1000.0})
        self.assertGreater(exec_p, 100.0)

        vol = SlippageModelFactory.create("volatility", volatility_factor=0.5)
        exec_v = vol.calculate_execution_price(100.0, 1.0, "BUY", {"high": 102.0, "low": 98.0})
        self.assertEqual(exec_v, 102.0)

    def test_spread_models(self) -> None:
        fixed = SpreadModelFactory.create("fixed", pips=1.5, point_value=0.0001)
        self.assertAlmostEqual(fixed.get_spread(None, 1.10), 0.00015)
        bid, ask = fixed.get_bid_ask(None, 1.10)
        self.assertAlmostEqual(ask - bid, 0.00015)

        broker_sp = SpreadModelFactory.create("broker", asset_symbol="EURUSD")
        self.assertAlmostEqual(broker_sp.get_spread(None, 1.10), 0.00010)

    def test_latency_models(self) -> None:
        exec_lat = LatencyModelFactory.create("execution", bar_delay=2)
        bars, ms = exec_lat.calculate_delay(None, 0)
        self.assertEqual(bars, 2)

        comp = LatencyModelFactory.create("composite", bar_delay=1, network_ms=20, broker_ms=30)
        c_bars, c_sec = comp.calculate_delay(None, 0)
        self.assertEqual(c_bars, 1)
        self.assertAlmostEqual(c_sec, 0.065)

    def test_order_manager(self) -> None:
        om = OrderManager()
        order = om.create_order(
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            created_bar_index=1,
        )
        self.assertEqual(order.status, OrderStatus.PENDING)
        self.assertTrue(order.is_active)

        om.update_order_fill(order.order_id, fill_qty=1.0, fill_price=1.1000)
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertFalse(order.is_active)

    def test_position_manager(self) -> None:
        pm = PositionManager()
        pos = pm.open_position(
            symbol="EURUSD",
            side=PositionSide.LONG,
            quantity=1.0,
            entry_price=1.1000,
            stop_loss=1.0900,
            take_profit=1.1200,
            breakeven_pips=10.0,
        )
        self.assertTrue(pos.is_open)
        self.assertEqual(len(pm.get_open_positions()), 1)

        # Scale in
        pos.scale_in(add_qty=1.0, fill_price=1.1050)
        self.assertEqual(pos.quantity, 2.0)
        self.assertEqual(pos.entry_price, 1.1025)

        # Test SL/TP trigger update
        exits = pm.update_positions_on_bar("EURUSD", 1.1025, 1.1250, 1.1050, 1.1200)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0][1], "TAKE_PROFIT")

    def test_portfolio_manager(self) -> None:
        port = PortfolioManager(initial_capital=100000.0, leverage=100.0)
        self.assertTrue(port.check_margin_availability(1000.0))
        self.assertEqual(port.calculate_required_margin(100000.0), 1000.0)

        # Process PnL
        port.process_realized_pnl(500.0)
        self.assertEqual(port.balance, 100500.0)

        state = port.update_state("2026-01-01", [])
        self.assertEqual(state.equity, 100500.0)
        self.assertFalse(port.check_stop_out(state))

    def test_execution_simulator(self) -> None:
        sim = ExecutionSimulator()
        om = OrderManager()
        pm = PositionManager()
        port = PortfolioManager(initial_capital=100000.0)

        om.create_order("EURUSD", OrderSide.BUY, OrderType.MARKET, 1.0, created_bar_index=0)
        bar_data = {"symbol": "EURUSD", "open": 1.1000, "high": 1.1050, "low": 1.0950, "close": 1.1020, "volume": 5000.0}

        exec_orders, aff_positions = sim.process_order_queues(bar_data, 0, om, pm, port)
        self.assertEqual(len(exec_orders), 1)
        self.assertEqual(len(aff_positions), 1)
        self.assertEqual(exec_orders[0].status, OrderStatus.FILLED)

    def test_equity_curve_and_trade_log(self) -> None:
        eq = EquityCurve()
        eq.add_point("2026-01-01 09:30", 100000.0, 100000.0)
        eq.add_point("2026-01-01 09:31", 100000.0, 101000.0, drawdown_amount=0.0, drawdown_pct=0.0)
        eq.add_point("2026-01-01 09:32", 100000.0, 99000.0, drawdown_amount=2000.0, drawdown_pct=1.98)

        df_eq = eq.to_dataframe()
        self.assertEqual(len(df_eq), 3)
        self.assertEqual(eq.calculate_peak_equity(), 101000.0)

        tl = TradeLog()
        tl.record_trade("POS-1", "EURUSD", "LONG", 1.0, "09:30", 1.1000, "09:32", 1.1100)
        self.assertEqual(len(tl.get_winning_trades()), 1)

    def test_metrics_and_statistics(self) -> None:
        eq = EquityCurve()
        eq.add_point("09:30", 100000.0, 100000.0)
        eq.add_point("09:31", 100000.0, 105000.0)

        tl = TradeLog()
        tl.record_trade("P1", "EURUSD", "LONG", 1.0, "09:30", 100.0, "09:31", 105.0)

        m = PerformanceMetrics.calculate_all(eq.to_dataframe(), tl.to_dataframe(), initial_capital=100000.0)
        self.assertEqual(m["net_profit"], 5000.0)
        self.assertEqual(m["total_trades"], 1)

        dist = BacktestStatistics.calculate_trade_distribution(tl.to_dataframe())
        self.assertIn("mean", dist)

        var = BacktestStatistics.calculate_value_at_risk(eq.get_returns_series())
        self.assertIn("historical_var_pct", var)

    def test_report_generator(self) -> None:
        engine = BacktestEngine()
        engine.load_dataset(self.dataset)
        engine.load_strategy(self.strategy)
        res = engine.start_simulation()

        reporter = ReportGenerator(res)
        out_dir = os.path.join(self.temp_dir, "reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["trade_log_csv"]))
        self.assertTrue(os.path.exists(paths["equity_curve_csv"]))

    def test_backtest_replay(self) -> None:
        replay = BacktestReplay()
        replay.add_snapshot(ReplaySnapshot(0, "09:30", {"close": 100.0}, balance=100000.0, equity=100000.0))
        replay.add_snapshot(ReplaySnapshot(1, "09:31", {"close": 101.0}, balance=100000.0, equity=101000.0))

        self.assertEqual(replay.total_steps, 2)
        snap = replay.step_forward()
        self.assertIsNotNone(snap)
        self.assertEqual(snap.step_index, 1)

        jump_snap = replay.jump_to_step(0)
        self.assertEqual(jump_snap.step_index, 0)

    def test_backtest_engine_and_runner(self) -> None:
        config = BacktestConfig(initial_capital=100000.0, default_trade_size=1.0)
        runner = BacktestRunner(config)

        res = runner.run_single(
            self.strategy,
            self.dataset,
            export_reports=True,
            output_dir=os.path.join(self.temp_dir, "runner_out"),
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.strategy_name, "DummyTestStrategy")
        self.assertGreater(res.execution_time_seconds, 0.0)

        # Batch runner test
        batch_results = runner.run_batch([self.strategy], [self.dataset])
        self.assertEqual(len(batch_results), 1)

        # Parameter grid test
        grid_results = runner.run_parameter_grid(DummyTestStrategy, {}, self.dataset)
        self.assertEqual(len(grid_results), 1)


if __name__ == "__main__":
    unittest.main()
