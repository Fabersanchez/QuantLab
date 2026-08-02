"""
QuantLab Master Monte Carlo & Robustness Engine.

Orchestrates multi-thousand iteration Monte Carlo simulations, statistical bootstrap sampling,
trade order permutations, extreme market stress testing, parameter sensitivity analysis,
distribution calculations, risk probabilities, confidence intervals, robustness scoring, and reporting.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestResult
from core.logger import get_logger
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy

from monte_carlo.scenario_generator import ScenarioType
from monte_carlo.simulation_runner import SimulationIterationResult, SimulationRunner
from monte_carlo.distribution_metrics import DistributionMetricsCalculator
from monte_carlo.probability_analysis import ProbabilityAnalyzer
from monte_carlo.confidence_intervals import ConfidenceIntervalCalculator
from monte_carlo.robustness_score import InstitutionalRobustnessScore
from monte_carlo.stress_testing import StressTestResult, StressTestRunner
from monte_carlo.sensitivity_analysis import SensitivityAnalyzer


logger = get_logger("MonteCarloEngine")


@dataclass
class MonteCarloConfig:
    """Configuration options for MonteCarloEngine execution."""

    iterations: int = 1000
    scenario_type: ScenarioType = ScenarioType.TRADE_PERMUTATION
    bootstrap_method: str = "random"
    block_size: int = 10
    ruin_threshold_pct: float = 50.0
    initial_capital: float = 100000.0
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    run_stress_tests: bool = True
    run_sensitivity: bool = False
    export_reports: bool = False


@dataclass
class MonteCarloResult:
    """Dataclass holding complete Monte Carlo analysis outputs."""

    strategy_name: str
    asset_symbol: str
    total_iterations: int
    iteration_results: List[SimulationIterationResult]
    distribution_metrics: Dict[str, float]
    probability_metrics: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    robustness_score: Dict[str, float]
    stress_test_results: Optional[List[StressTestResult]] = None
    sensitivity_results: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0


class MonteCarloEngine:
    """Master Institutional Monte Carlo Engine."""

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        """Initialize MonteCarloEngine."""
        self.config = config or MonteCarloConfig()
        self._backtest_result: Optional[BacktestResult] = None
        self._dataset: Optional[MarketDataset] = None
        self._strategy: Optional[BaseStrategy] = None
        self._is_running: bool = False

    def load_backtest_result(self, result: BacktestResult) -> None:
        """Load completed BacktestResult object."""
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be an instance of BacktestResult.")
        self._backtest_result = result
        logger.info(f"Loaded BacktestResult for Monte Carlo: Strategy='{result.strategy_name}'")

    def load_dataset_and_strategy(self, dataset: MarketDataset, strategy: BaseStrategy) -> None:
        """Load raw dataset and strategy."""
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be an instance of MarketDataset.")
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must inherit from BaseStrategy.")
        self._dataset = dataset
        self._strategy = strategy
        logger.info(f"Loaded dataset & strategy for Monte Carlo: '{strategy.metadata().name}'")

    def stop_monte_carlo(self) -> None:
        """Halt active Monte Carlo simulation."""
        self._is_running = False
        logger.warning("Monte Carlo simulation stop requested.")

    def start_monte_carlo(
        self, walk_forward_metrics: Optional[Dict[str, float]] = None
    ) -> MonteCarloResult:
        """Execute full Monte Carlo simulation suite.

        Returns:
            MonteCarloResult dataclass.
        """
        start_time = time.time()
        self._is_running = True

        strategy_name = "CustomStrategy"
        asset_symbol = "GENERIC"
        trades_list: List[float] = []

        # Extract trades from loaded BacktestResult or dataset/strategy
        if self._backtest_result:
            strategy_name = self._backtest_result.strategy_name
            asset_symbol = self._backtest_result.asset_symbol
            trade_df = self._backtest_result.trade_log.to_dataframe()
            if not trade_df.empty and "net_pnl" in trade_df.columns:
                trades_list = trade_df["net_pnl"].tolist()
        elif self._dataset and self._strategy:
            strategy_name = self._strategy.metadata().name
            asset_symbol = self._dataset.metadata.asset
            from backtesting.backtest_engine import BacktestEngine
            eng = BacktestEngine(BacktestConfig(initial_capital=self.config.initial_capital))
            eng.load_dataset(self._dataset)
            eng.load_strategy(self._strategy)
            bt_res = eng.start_simulation()
            self._backtest_result = bt_res
            trade_df = bt_res.trade_log.to_dataframe()
            if not trade_df.empty and "net_pnl" in trade_df.columns:
                trades_list = trade_df["net_pnl"].tolist()

        if not trades_list:
            logger.warning("No trade log entries found. Synthesizing baseline zero trade entry.")
            trades_list = [0.0]

        logger.info(
            f"Starting Monte Carlo simulation ({self.config.iterations:,} iterations): "
            f"Strategy='{strategy_name}', Scenario='{self.config.scenario_type.value}'..."
        )

        # 1. Execute Simulation Runner across N iterations
        runner = SimulationRunner(
            initial_capital=self.config.initial_capital,
            ruin_threshold_pct=self.config.ruin_threshold_pct,
        )

        sim_iterations = runner.run_simulations_from_trades(
            trades=trades_list,
            n_iterations=self.config.iterations,
            scenario_type=self.config.scenario_type,
            bootstrap_method=self.config.bootstrap_method,
            block_size=self.config.block_size,
        )

        # 2. Compute Distribution Metrics
        dist_metrics = DistributionMetricsCalculator.calculate_distribution_metrics(sim_iterations)

        # 3. Compute Probabilistic Risk Metrics
        prob_metrics = ProbabilityAnalyzer.calculate_all_probabilities(sim_iterations)

        # 4. Compute Confidence Intervals (90%, 95%, 99%)
        conf_intervals = ConfidenceIntervalCalculator.calculate_confidence_intervals(
            sim_iterations, self.config.confidence_levels
        )

        # 5. Stress Testing (if enabled and dataset/strategy available)
        stress_results = None
        if self.config.run_stress_tests and self._dataset and self._strategy:
            logger.info("Executing extreme market stress tests...")
            stress_results = StressTestRunner.run_all_stress_tests(
                strategy=self._strategy,
                dataset=self._dataset,
                config=BacktestConfig(initial_capital=self.config.initial_capital),
            )

        # 6. Sensitivity Analysis (if enabled)
        sens_results = None
        elasticity_score = None
        if self.config.run_sensitivity and self._dataset and self._strategy:
            logger.info("Executing parameter sensitivity analysis...")
            sens_results = SensitivityAnalyzer.analyze_execution_cost_sensitivity(
                strategy=self._strategy,
                dataset=self._dataset,
            )
            if "spread" in sens_results:
                elasticity_score = SensitivityAnalyzer.calculate_elasticity_score(sens_results["spread"])

        # 7. Compute Composite Institutional Robustness Score
        robustness_score = InstitutionalRobustnessScore.calculate_score(
            monte_carlo_results=sim_iterations,
            walk_forward_metrics=walk_forward_metrics,
            sensitivity_elasticity=elasticity_score,
        )

        exec_duration = time.time() - start_time
        self._is_running = False

        logger.info(
            f"Monte Carlo simulation completed cleanly in {exec_duration:.2f}s! "
            f"Robustness Score={robustness_score.get('institutional_robustness_score', 0):.1f}/100"
        )

        return MonteCarloResult(
            strategy_name=strategy_name,
            asset_symbol=asset_symbol,
            total_iterations=self.config.iterations,
            iteration_results=sim_iterations,
            distribution_metrics=dist_metrics,
            probability_metrics=prob_metrics,
            confidence_intervals=conf_intervals,
            robustness_score=robustness_score,
            stress_test_results=stress_results,
            sensitivity_results=sens_results,
            execution_time_seconds=exec_duration,
        )
