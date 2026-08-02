"""
QuantLab High-Level Backtest Runner and Orchestrator.

Provides simplified execution workflows for single strategy runs, batch multi-strategy multi-asset runs,
hyperparameter grid sweeps, and automated multi-format report generation.
"""

import itertools
import os
from typing import Any, Dict, List, Optional, Type
import pandas as pd

from core.logger import get_logger
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.report_generator import ReportGenerator


logger = get_logger("BacktestRunner")


class BacktestRunner:
    """High-Level Backtest Orchestrator and Parameter Sweep Engine."""

    def __init__(self, engine_config: Optional[BacktestConfig] = None) -> None:
        """Initialize BacktestRunner."""
        self.config = engine_config or BacktestConfig()

    def run_single(
        self,
        strategy: BaseStrategy,
        dataset: MarketDataset,
        config: Optional[BacktestConfig] = None,
        export_reports: bool = False,
        output_dir: Optional[str] = None,
    ) -> BacktestResult:
        """Run single backtest simulation for strategy and dataset.

        Args:
            strategy: BaseStrategy instance.
            dataset: MarketDataset instance.
            config: Optional BacktestConfig overrides.
            export_reports: If True, exports reports to output_dir.
            output_dir: Directory path for report exports.

        Returns:
            BacktestResult dataclass.
        """
        active_config = config or self.config
        engine = BacktestEngine(config=active_config)
        engine.load_dataset(dataset)
        engine.load_strategy(strategy)

        result = engine.start_simulation()

        if export_reports and output_dir:
            reporter = ReportGenerator(result)
            exported = reporter.export_all(output_dir)
            logger.info(f"Exported backtest reports to '{output_dir}': {list(exported.keys())}")

        return result

    def run_batch(
        self,
        strategies: List[BaseStrategy],
        datasets: List[MarketDataset],
        config: Optional[BacktestConfig] = None,
    ) -> List[BacktestResult]:
        """Run batch simulation across multiple strategies and datasets.

        Returns:
            List of BacktestResult objects for each pair combination.
        """
        results: List[BacktestResult] = []
        active_config = config or self.config

        for strat in strategies:
            for ds in datasets:
                logger.info(f"Running batch pair: Strategy='{strat.metadata().name}', Asset='{ds.metadata.asset}'")
                res = self.run_single(strat, ds, config=active_config, export_reports=False)
                results.append(res)

        return results

    def run_parameter_grid(
        self,
        strategy_cls: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        dataset: MarketDataset,
        config: Optional[BacktestConfig] = None,
    ) -> List[BacktestResult]:
        """Run parameter grid search sweep across hyperparameter variations.

        Args:
            strategy_cls: BaseStrategy class type to instantiate.
            param_grid: Dictionary mapping parameter names to lists of values to test.
            dataset: Target MarketDataset.
            config: Engine configuration.

        Returns:
            List of BacktestResult objects corresponding to parameter grid iterations.
        """
        active_config = config or self.config
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        results: List[BacktestResult] = []
        logger.info(f"Starting parameter grid sweep: {len(combinations)} parameter combinations...")

        for combo in combinations:
            params = dict(zip(param_names, combo))
            strat_instance = strategy_cls(params=params)
            res = self.run_single(strat_instance, dataset, config=active_config, export_reports=False)
            results.append(res)

        return results
