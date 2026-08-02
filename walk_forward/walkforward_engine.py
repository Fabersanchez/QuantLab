"""
QuantLab Master Walk Forward Optimization Engine.

Orchestrates strategy windowing generation, in-sample hyperparameter optimization, out-of-sample validation,
result stitching, robustness metric calculation, and reporting.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from backtesting.backtest_engine import BacktestConfig
from core.logger import get_logger
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy

from walk_forward.window_generator import BaseWindowGenerator, WindowGeneratorFactory, WindowSplit
from walk_forward.optimizer_interface import BaseOptimizerAdapter, OptimizerAdapterFactory
from walk_forward.validation_runner import ValidationRunner, ValidationStepResult
from walk_forward.window_statistics import WindowStatisticsCalculator
from walk_forward.robustness_metrics import RobustnessMetricsCalculator
from walk_forward.efficiency import EfficiencyAnalyzer


logger = get_logger("WalkForwardEngine")


@dataclass
class WalkForwardConfig:
    """Configuration options for WalkForwardEngine execution."""

    train_bars: int = 252
    val_bars: int = 63
    step_bars: Optional[int] = None
    window_type: str = "rolling"  # 'rolling', 'expanding', 'anchored', 'sliding', 'custom'
    optimizer_type: str = "grid"  # 'grid', 'random', 'optuna', 'bayesian', 'genetic', 'pso'
    metric_target: str = "sharpe_ratio"
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    initial_capital: float = 100000.0
    leverage: float = 100.0
    export_reports: bool = False
    custom_window_bounds: Optional[List[Any]] = None


@dataclass
class WalkForwardResult:
    """Dataclass holding complete Walk Forward analysis outputs."""

    strategy_name: str
    asset_symbol: str
    window_splits: List[WindowSplit]
    window_results: List[ValidationStepResult]
    concatenated_oos_equity: pd.DataFrame
    concatenated_oos_trades: pd.DataFrame
    robustness_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, Any]
    statistics_summary: Dict[str, Any]
    execution_time_seconds: float = 0.0


class WalkForwardEngine:
    """Master Institutional Walk Forward Engine."""

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        """Initialize WalkForwardEngine."""
        self.config = config or WalkForwardConfig()
        self._dataset: Optional[MarketDataset] = None
        self._strategy: Optional[BaseStrategy] = None
        self._window_generator: Optional[BaseWindowGenerator] = None
        self._optimizer: Optional[BaseOptimizerAdapter] = None
        self._is_running: bool = False

    def load_dataset(self, dataset: MarketDataset) -> None:
        """Load market dataset container."""
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be an instance of MarketDataset.")
        self._dataset = dataset
        logger.info(f"Loaded WFA dataset: Asset='{dataset.metadata.asset}', Rows={dataset.rows}")

    def load_strategy(self, strategy: BaseStrategy) -> None:
        """Load quantitative strategy."""
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must inherit from BaseStrategy.")
        self._strategy = strategy
        logger.info(f"Loaded WFA strategy: '{strategy.metadata().name}'")

    def set_param_grid(self, param_grid: Dict[str, List[Any]]) -> None:
        """Set hyperparameter search grid."""
        self.config.param_grid = param_grid

    def set_window_generator(self, generator: BaseWindowGenerator) -> None:
        """Set custom window generator."""
        self._window_generator = generator

    def set_optimizer(self, optimizer: BaseOptimizerAdapter) -> None:
        """Set custom optimizer adapter."""
        self._optimizer = optimizer

    def stop_walkforward(self) -> None:
        """Halt active Walk Forward execution."""
        self._is_running = False
        logger.warning("Walk Forward execution stop requested.")

    def start_walkforward(self) -> WalkForwardResult:
        """Execute complete Walk Forward Optimization & Validation sequence.

        Returns:
            WalkForwardResult dataclass.
        """
        if self._dataset is None:
            raise RuntimeError("No MarketDataset loaded. Call load_dataset() first.")
        if self._strategy is None:
            raise RuntimeError("No BaseStrategy loaded. Call load_strategy() first.")

        start_time = time.time()
        self._is_running = True

        strat_cls = type(self._strategy)
        strat_name = self._strategy.metadata().name
        asset = self._dataset.metadata.asset
        df = self._dataset.data

        logger.info(
            f"Starting Walk Forward Engine: Strategy='{strat_name}', Asset='{asset}', WindowType='{self.config.window_type}'..."
        )

        # 1. Instantiate Window Generator
        if self._window_generator is None:
            if self.config.window_type == "custom" and self.config.custom_window_bounds:
                self._window_generator = WindowGeneratorFactory.create(
                    "custom", explicit_bounds=self.config.custom_window_bounds
                )
            else:
                self._window_generator = WindowGeneratorFactory.create(
                    self.config.window_type,
                    train_bars=self.config.train_bars,
                    val_bars=self.config.val_bars,
                    step_bars=self.config.step_bars,
                )

        windows = self._window_generator.generate_windows(df)
        logger.info(f"Generated {len(windows)} Walk Forward window splits.")

        # 2. Instantiate Optimizer Adapter
        if self._optimizer is None:
            self._optimizer = OptimizerAdapterFactory.create(self.config.optimizer_type)

        # 3. Execute Validation Runner
        val_runner = ValidationRunner(optimizer_adapter=self._optimizer)

        engine_backtest_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            leverage=self.config.leverage,
        )

        step_results = val_runner.run_validation_sequence(
            strategy_cls=strat_cls,
            param_grid=self.config.param_grid,
            dataset=self._dataset,
            windows=windows,
            metric_target=self.config.metric_target,
            config=engine_backtest_config,
        )

        # 4. Stitch Concatenated OOS Equity Curve & Trades
        concat_equity = self._stitch_oos_equity(step_results)
        concat_trades = self._stitch_oos_trades(step_results)

        # 5. Compute Robustness, Efficiency, and Statistics Metrics
        robustness = RobustnessMetricsCalculator.calculate_all(step_results, len(df))
        efficiency = EfficiencyAnalyzer.analyze_efficiency(
            step_results, concat_equity, initial_capital=self.config.initial_capital
        )
        stats_summary = WindowStatisticsCalculator.compute_aggregate_statistics(step_results)

        exec_duration = time.time() - start_time
        self._is_running = False

        logger.info(
            f"Walk Forward completed cleanly in {exec_duration:.2f}s! WFE={robustness.get('walk_forward_efficiency_pct', 0):.1f}%"
        )

        return WalkForwardResult(
            strategy_name=strat_name,
            asset_symbol=asset,
            window_splits=windows,
            window_results=step_results,
            concatenated_oos_equity=concat_equity,
            concatenated_oos_trades=concat_trades,
            robustness_metrics=robustness,
            efficiency_metrics=efficiency,
            statistics_summary=stats_summary,
            execution_time_seconds=exec_duration,
        )

    def _stitch_oos_equity(self, step_results: List[ValidationStepResult]) -> pd.DataFrame:
        """Stitch together Out-of-Sample equity curves into a single continuous DataFrame."""
        if not step_results:
            return pd.DataFrame()

        dfs = []
        running_equity = self.config.initial_capital

        for s in step_results:
            eq_df = s.oos_equity_df.copy()
            if not eq_df.empty and "equity" in eq_df.columns:
                start_eq = eq_df["equity"].iloc[0]
                if start_eq > 0:
                    mult = running_equity / start_eq
                    eq_df["equity"] = eq_df["equity"] * mult
                    if "balance" in eq_df.columns:
                        eq_df["balance"] = eq_df["balance"] * mult

                running_equity = eq_df["equity"].iloc[-1]
                dfs.append(eq_df)

        if not dfs:
            return pd.DataFrame()

        stitched = pd.concat(dfs)
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        return stitched

    def _stitch_oos_trades(self, step_results: List[ValidationStepResult]) -> pd.DataFrame:
        """Stitch together Out-of-Sample trade logs."""
        dfs = [s.oos_trades_df for s in step_results if not s.oos_trades_df.empty]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
