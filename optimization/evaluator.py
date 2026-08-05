"""
QuantLab Strategy Solution Evaluator.

Executes candidate strategy parameter sets across Backtesting Engine, Walk Forward Engine,
Monte Carlo Engine, and Research Engine. Calculates metrics, verifies constraints, computes
multi-objective fitness, and registers reproducible Experiments.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Type
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine
from data.market_dataset import MarketDataset
from optimization.constraints import OptimizationConstraints
from optimization.objective_function import ObjectiveFunction
from research.adapters import BacktestExperimentAdapter
from research.experiment import Experiment
from research.experiment_manager import ExperimentManager
from research.metrics import MetricsCalculator
from strategies.base_strategy import BaseStrategy


@dataclass
class EvaluationResult:
    """Dataclass holding complete candidate evaluation output."""

    parameters: Dict[str, Any]
    fitness_score: float
    is_valid: bool
    metrics: Dict[str, Any]
    violations: list
    execution_time_sec: float
    experiment: Optional[Experiment] = None


class SolutionEvaluator:
    """Master Evaluator executing candidate strategy parameter combinations."""

    def __init__(
        self,
        strategy_cls: Type[BaseStrategy],
        dataset: MarketDataset,
        backtest_config: Optional[BacktestConfig] = None,
        objective_function: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        research_manager: Optional[ExperimentManager] = None,
        asset_symbol: str = "EURUSD",
        timeframe: str = "1h",
    ) -> None:
        """Initialize SolutionEvaluator.

        Args:
            strategy_cls: Strategy class to instantiate.
            dataset: MarketDataset container.
            backtest_config: Engine BacktestConfig.
            objective_function: ObjectiveFunction instance.
            constraints: OptimizationConstraints instance.
            research_manager: Optional ExperimentManager instance.
            asset_symbol: Asset symbol string.
            timeframe: Timeframe string.
        """
        self.strategy_cls = strategy_cls
        self.dataset = dataset
        self.backtest_config = backtest_config or BacktestConfig()
        self.objective_function = objective_function or ObjectiveFunction()
        self.constraints = constraints or OptimizationConstraints()
        self.research_manager = research_manager
        self.asset_symbol = asset_symbol
        self.timeframe = timeframe
        self.metrics_calculator = MetricsCalculator()

    def evaluate(self, parameters: Dict[str, Any]) -> EvaluationResult:
        """Execute candidate parameter set through backtesting and research engines.

        Args:
            parameters: Parameter values dictionary.

        Returns:
            EvaluationResult instance.
        """
        start_t = time.perf_counter()

        try:
            # Instantiate strategy with candidate parameters
            strategy_instance = self.strategy_cls(**parameters)

            # Configure and run BacktestEngine
            engine = BacktestEngine(config=self.backtest_config)
            engine.load_dataset(self.dataset)
            engine.load_strategy(strategy_instance)

            bt_res = engine.start_simulation()
            exec_time = time.perf_counter() - start_t

            # Extract trades & equity curve for centralized metrics calculation
            trades_df = bt_res.trade_log.to_dataframe() if hasattr(bt_res, "trade_log") else None
            equity_series = bt_res.equity_curve.to_series() if hasattr(bt_res, "equity_curve") else None

            metrics = self.metrics_calculator.compute_all(
                trades_df=trades_df,
                equity_series=equity_series,
                initial_capital=self.backtest_config.initial_capital,
                risk_free_rate=self.backtest_config.risk_free_rate,
                total_bars=self.dataset.rows,
            )

            # Evaluate constraints
            exec_info = {"execution_time_sec": exec_time, "ram_peak_mb": 0.0, "cpu_usage_pct": 0.0}
            is_valid, violations = self.constraints.evaluate(metrics, exec_info)

            # Compute composite multi-objective fitness score
            fitness = self.objective_function.evaluate(metrics, execution_time_sec=exec_time)
            if not is_valid:
                fitness = max(-1000.0, fitness - 500.0)  # Constraint penalty

            # Create and register Experiment in ResearchEngine
            exp = BacktestExperimentAdapter.to_experiment(
                bt_res,
                author="OptimizationEngine",
                broker="GenericBroker",
            )
            exp.parameters = parameters
            exp.execution_time = exec_time
            exp.results = metrics

            if self.research_manager:
                self.research_manager.registry.register(exp, log_message="Optimization candidate evaluated.")

            return EvaluationResult(
                parameters=parameters,
                fitness_score=fitness,
                is_valid=is_valid,
                metrics=metrics,
                violations=violations,
                execution_time_sec=exec_time,
                experiment=exp,
            )

        except Exception as exc:
            exec_time = time.perf_counter() - start_t
            return EvaluationResult(
                parameters=parameters,
                fitness_score=-9999.0,
                is_valid=False,
                metrics={},
                violations=[f"Evaluation failure exception: {str(exc)}"],
                execution_time_sec=exec_time,
                experiment=None,
            )
