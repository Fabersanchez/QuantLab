"""
QuantLab Master Institutional Backtest Engine.

Orchestrates strategy simulation, dataset iteration, realistic execution,
portfolio margin/equity updates, position lifecycle management, performance metric calculation,
and simulation replay recording.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from core.logger import get_logger
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy

from backtesting.commission_model import BaseCommissionModel, FixedCommissionModel
from backtesting.slippage_model import BaseSlippageModel, FixedSlippageModel
from backtesting.spread_model import BaseSpreadModel, FixedSpreadModel
from backtesting.latency_model import BaseLatencyModel, ExecutionDelayModel
from backtesting.order_manager import OrderManager, OrderSide, OrderType
from backtesting.position_manager import PositionManager, PositionSide
from backtesting.portfolio_manager import PortfolioManager
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.equity_curve import EquityCurve
from backtesting.trade_log import TradeLog
from backtesting.metrics import PerformanceMetrics
from backtesting.statistics import BacktestStatistics
from backtesting.replay import BacktestReplay, ReplaySnapshot


logger = get_logger("BacktestEngine")


@dataclass
class BacktestConfig:
    """Configuration specification for BacktestEngine execution."""

    initial_capital: float = 100000.0
    leverage: float = 100.0
    margin_call_pct: float = 100.0
    stop_out_pct: float = 50.0
    point_value: float = 0.0001
    warm_up_bars: int = 20
    enable_replay: bool = True
    risk_free_rate: float = 0.0
    default_trade_size: float = 1.0


@dataclass
class BacktestResult:
    """Dataclass containing complete backtest outputs and analytical containers."""

    strategy_name: str
    asset_symbol: str
    timeframe: str
    config: BacktestConfig
    metrics: Dict[str, Any]
    statistics: Dict[str, Any]
    equity_curve: EquityCurve
    trade_log: TradeLog
    replay: BacktestReplay
    execution_time_seconds: float = 0.0


class BacktestEngine:
    """Master Institutional Backtesting Engine."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialize BacktestEngine.

        Args:
            config: Optional BacktestConfig instance.
        """
        self.config = config or BacktestConfig()

        # Component models
        self.commission_model: BaseCommissionModel = FixedCommissionModel(0.0)
        self.slippage_model: BaseSlippageModel = FixedSlippageModel(0.0)
        self.spread_model: BaseSpreadModel = FixedSpreadModel(0.0)
        self.latency_model: BaseLatencyModel = ExecutionDelayModel(0)

        # Dataset and Strategy
        self._dataset: Optional[MarketDataset] = None
        self._strategy: Optional[BaseStrategy] = None
        self._is_running: bool = False

    def load_dataset(self, dataset: MarketDataset) -> None:
        """Load market dataset container into backtest engine."""
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be an instance of MarketDataset.")
        self._dataset = dataset
        logger.info(
            f"Dataset loaded: Asset='{dataset.metadata.asset}', "
            f"Timeframe='{dataset.metadata.timeframe}', Rows={dataset.rows}"
        )

    def load_strategy(self, strategy: BaseStrategy) -> None:
        """Load quantitative strategy into backtest engine."""
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must inherit from BaseStrategy.")
        self._strategy = strategy
        logger.info(f"Strategy loaded: '{strategy.metadata().name}'")

    def set_commission_model(self, model: BaseCommissionModel) -> None:
        """Set active commission model."""
        self.commission_model = model

    def set_slippage_model(self, model: BaseSlippageModel) -> None:
        """Set active slippage model."""
        self.slippage_model = model

    def set_spread_model(self, model: BaseSpreadModel) -> None:
        """Set active spread model."""
        self.spread_model = model

    def set_latency_model(self, model: BaseLatencyModel) -> None:
        """Set active latency model."""
        self.latency_model = model

    def stop_simulation(self) -> None:
        """Emergency stop active simulation."""
        self._is_running = False
        logger.warning("Simulation stop requested by user.")

    def start_simulation(self) -> BacktestResult:
        """Execute full backtest simulation run.

        Returns:
            BacktestResult dataclass containing metrics, equity curve, trade log, and replay snapshots.
        """
        if self._dataset is None:
            raise RuntimeError("No MarketDataset loaded. Call load_dataset() first.")
        if self._strategy is None:
            raise RuntimeError("No BaseStrategy loaded. Call load_strategy() first.")

        start_time = time.time()
        self._is_running = True

        df = self._dataset.data.copy()
        asset = self._dataset.metadata.asset
        timeframe = self._dataset.metadata.timeframe
        strategy_name = self._strategy.metadata().name

        logger.info(f"Starting simulation: Strategy='{strategy_name}', Asset='{asset}'...")

        # 1. Strategy execution to obtain indicators & signals
        prepared_df = self._strategy.execute(df)

        if "signal" not in prepared_df.columns:
            prepared_df["signal"] = 0

        # 2. Instantiate managers
        order_mgr = OrderManager()
        pos_mgr = PositionManager()
        portfolio_mgr = PortfolioManager(
            initial_capital=self.config.initial_capital,
            leverage=self.config.leverage,
            margin_call_pct=self.config.margin_call_pct,
            stop_out_pct=self.config.stop_out_pct,
        )

        exec_sim = ExecutionSimulator(
            commission_model=self.commission_model,
            slippage_model=self.slippage_model,
            spread_model=self.spread_model,
            latency_model=self.latency_model,
            point_value=self.config.point_value,
        )

        equity_curve = EquityCurve()
        trade_log = TradeLog()
        replay = BacktestReplay()

        open_col = "open" if "open" in prepared_df.columns else "close"
        high_col = "high" if "high" in prepared_df.columns else "close"
        low_col = "low" if "low" in prepared_df.columns else "close"
        close_col = "close"
        vol_col = "volume" if "volume" in prepared_df.columns else None

        # Warm up bars
        warm_up = max(0, self.config.warm_up_bars)

        # 3. Bar-by-bar simulation loop
        for idx in range(len(prepared_df)):
            if not self._is_running:
                logger.warning(f"Simulation halted prematurely at bar index {idx}.")
                break

            row = prepared_df.iloc[idx]
            timestamp = row.name if not isinstance(row.name, int) else row.get("timestamp", idx)

            bar_open = float(row[open_col])
            bar_high = float(row[high_col])
            bar_low = float(row[low_col])
            bar_close = float(row[close_col])
            bar_vol = float(row[vol_col]) if vol_col and vol_col in row else 1000000.0
            signal_val = int(row["signal"]) if not pd.isna(row["signal"]) else 0

            bar_dict = {
                "symbol": asset,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": bar_vol,
                "timestamp": timestamp,
            }

            # A. Update existing positions on current bar (check SL / TP / Breakeven / Trailing Stop)
            triggered_exits = pos_mgr.update_positions_on_bar(
                symbol=asset,
                open_p=bar_open,
                high_p=bar_high,
                low_p=bar_low,
                close_p=bar_close,
                point_value=self.config.point_value,
                timestamp=timestamp,
            )

            for pos, exit_reason, exit_price in triggered_exits:
                closed_pos = pos_mgr.close_position(
                    position_id=pos.position_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    closed_at=timestamp,
                )
                portfolio_mgr.process_realized_pnl(closed_pos.realized_pnl)

                # Record trade log
                trade_log.record_trade(
                    position_id=closed_pos.position_id,
                    symbol=closed_pos.symbol,
                    side=closed_pos.side.value,
                    quantity=closed_pos.quantity,
                    entry_time=closed_pos.opened_at,
                    entry_price=closed_pos.entry_price,
                    exit_time=closed_pos.closed_at,
                    exit_price=closed_pos.exit_price,
                    commission=closed_pos.total_commission,
                    slippage=closed_pos.total_slippage,
                    exit_reason=closed_pos.exit_reason or exit_reason,
                    holding_bars=idx - 0,
                )

            # B. Process pending orders via ExecutionSimulator
            exec_sim.process_order_queues(
                bar_data=bar_dict,
                bar_index=idx,
                order_manager=order_mgr,
                position_manager=pos_mgr,
                portfolio_manager=portfolio_mgr,
            )

            # C. Evaluate strategy signal generation (after warm-up period)
            if idx >= warm_up and signal_val != 0:
                current_open_positions = pos_mgr.get_open_positions(asset)

                if signal_val > 0:  # LONG Signal
                    # Close any open short positions first
                    for short_p in [p for p in current_open_positions if p.side == PositionSide.SHORT]:
                        closed_pos = pos_mgr.close_position(
                            short_p.position_id, exit_price=bar_close, exit_reason="REVERSAL_SIGNAL", closed_at=timestamp
                        )
                        portfolio_mgr.process_realized_pnl(closed_pos.realized_pnl)
                        trade_log.record_trade(
                            position_id=closed_pos.position_id,
                            symbol=closed_pos.symbol,
                            side=closed_pos.side.value,
                            quantity=closed_pos.quantity,
                            entry_time=closed_pos.opened_at,
                            entry_price=closed_pos.entry_price,
                            exit_time=closed_pos.closed_at,
                            exit_price=closed_pos.exit_price,
                            commission=closed_pos.total_commission,
                            slippage=closed_pos.total_slippage,
                            exit_reason=closed_pos.exit_reason,
                        )

                    # Create BUY Market Order if no active LONG position
                    if not any(p.side == PositionSide.LONG for p in pos_mgr.get_open_positions(asset)):
                        order_mgr.create_order(
                            symbol=asset,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=self.config.default_trade_size,
                            created_at=timestamp,
                            created_bar_index=idx,
                        )

                elif signal_val < 0:  # SHORT Signal
                    # Close any open long positions first
                    for long_p in [p for p in current_open_positions if p.side == PositionSide.LONG]:
                        closed_pos = pos_mgr.close_position(
                            long_p.position_id, exit_price=bar_close, exit_reason="REVERSAL_SIGNAL", closed_at=timestamp
                        )
                        portfolio_mgr.process_realized_pnl(closed_pos.realized_pnl)
                        trade_log.record_trade(
                            position_id=closed_pos.position_id,
                            symbol=closed_pos.symbol,
                            side=closed_pos.side.value,
                            quantity=closed_pos.quantity,
                            entry_time=closed_pos.opened_at,
                            entry_price=closed_pos.entry_price,
                            exit_time=closed_pos.closed_at,
                            exit_price=closed_pos.exit_price,
                            commission=closed_pos.total_commission,
                            slippage=closed_pos.total_slippage,
                            exit_reason=closed_pos.exit_reason,
                        )

                    # Create SELL Market Order if no active SHORT position
                    if not any(p.side == PositionSide.SHORT for p in pos_mgr.get_open_positions(asset)):
                        order_mgr.create_order(
                            symbol=asset,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=self.config.default_trade_size,
                            created_at=timestamp,
                            created_bar_index=idx,
                        )

            # D. Update Portfolio state
            state = portfolio_mgr.update_state(timestamp, pos_mgr.get_open_positions(asset))

            # E. Forced liquidation check (Stop Out)
            if portfolio_mgr.check_stop_out(state):
                logger.warning(f"MARGIN STOP OUT TRIGGERED at bar {idx}! Liquidating all open positions.")
                for open_p in pos_mgr.get_open_positions(asset):
                    closed_pos = pos_mgr.close_position(
                        open_p.position_id, exit_price=bar_close, exit_reason="STOP_OUT_LIQUIDATION", closed_at=timestamp
                    )
                    portfolio_mgr.process_realized_pnl(closed_pos.realized_pnl)
                    trade_log.record_trade(
                        position_id=closed_pos.position_id,
                        symbol=closed_pos.symbol,
                        side=closed_pos.side.value,
                        quantity=closed_pos.quantity,
                        entry_time=closed_pos.opened_at,
                        entry_price=closed_pos.entry_price,
                        exit_time=closed_pos.closed_at,
                        exit_price=closed_pos.exit_price,
                        commission=closed_pos.total_commission,
                        slippage=closed_pos.total_slippage,
                        exit_reason="STOP_OUT_LIQUIDATION",
                    )
                state = portfolio_mgr.update_state(timestamp, [])

            # F. Record Equity Curve Point
            equity_curve.add_point(
                timestamp=timestamp,
                balance=state.balance,
                equity=state.equity,
                margin_used=state.margin_used,
                free_margin=state.free_margin,
                drawdown_amount=state.drawdown_amount,
                drawdown_pct=state.drawdown_pct,
                open_positions_count=len(pos_mgr.get_open_positions(asset)),
            )

            # G. Record Replay Snapshot if enabled
            if self.config.enable_replay:
                replay.add_snapshot(
                    ReplaySnapshot(
                        step_index=idx,
                        timestamp=timestamp,
                        bar_data=bar_dict,
                        open_orders_count=len(order_mgr.get_open_orders(asset)),
                        open_positions_count=len(pos_mgr.get_open_positions(asset)),
                        closed_positions_count=len(pos_mgr.get_closed_positions()),
                        balance=state.balance,
                        equity=state.equity,
                        drawdown_pct=state.drawdown_pct,
                        signal=signal_val,
                    )
                )

        # 4. Close any remaining open positions at final bar
        final_row = prepared_df.iloc[-1]
        final_close = float(final_row[close_col])
        final_ts = final_row.name if not isinstance(final_row.name, int) else final_row.get("timestamp", len(prepared_df)-1)

        for open_p in pos_mgr.get_open_positions(asset):
            closed_pos = pos_mgr.close_position(
                open_p.position_id, exit_price=final_close, exit_reason="SIMULATION_END", closed_at=final_ts
            )
            portfolio_mgr.process_realized_pnl(closed_pos.realized_pnl)
            trade_log.record_trade(
                position_id=closed_pos.position_id,
                symbol=closed_pos.symbol,
                side=closed_pos.side.value,
                quantity=closed_pos.quantity,
                entry_time=closed_pos.opened_at,
                entry_price=closed_pos.entry_price,
                exit_time=closed_pos.closed_at,
                exit_price=closed_pos.exit_price,
                commission=closed_pos.total_commission,
                slippage=closed_pos.total_slippage,
                exit_reason="SIMULATION_END",
            )

        # 5. Compute final metrics & statistics
        eq_df = equity_curve.to_dataframe()
        trd_df = trade_log.to_dataframe()

        metrics = PerformanceMetrics.calculate_all(
            equity_df=eq_df,
            trade_df=trd_df,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
        )

        trade_dist = BacktestStatistics.calculate_trade_distribution(trd_df)
        seasonality = BacktestStatistics.generate_seasonality_matrix(eq_df)
        holding_stats = BacktestStatistics.calculate_holding_duration_stats(trd_df)
        long_short_stats = BacktestStatistics.calculate_long_vs_short_breakdown(trd_df)
        var_stats = BacktestStatistics.calculate_value_at_risk(equity_curve.get_returns_series())

        statistics = {
            "trade_distribution": trade_dist,
            "holding_stats": holding_stats,
            "long_short": long_short_stats,
            "value_at_risk": var_stats,
            "seasonality_matrix": seasonality.to_dict() if not seasonality.empty else {},
        }

        exec_duration = time.time() - start_time
        self._is_running = False
        logger.info(f"Simulation completed cleanly in {exec_duration:.3f} seconds.")

        return BacktestResult(
            strategy_name=strategy_name,
            asset_symbol=asset,
            timeframe=timeframe,
            config=self.config,
            metrics=metrics,
            statistics=statistics,
            equity_curve=equity_curve,
            trade_log=trade_log,
            replay=replay,
            execution_time_seconds=exec_duration,
        )
