"""
QuantLab Master Strategy Engine.

Main orchestrator managing strategy registration, execution, enabling, disabling,
building, validation, export, and optimization in QuantLab.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.strategy_metadata import StrategyMetadata
from strategies.strategy_registry import StrategyRegistry
from strategies.strategy_validator import StrategyValidator, StrategyValidationReport
from strategies.strategy_builder import StrategyBuilder
from strategies.strategy_exporter import StrategyExporter
from strategies.strategy_pipeline import StrategyPipeline
from strategies.strategy_optimizer import OptimizerFactory, BaseOptimizer, OptimizationResult


class StrategyEngine:
    """Master Quantitative Strategy Engine."""

    def __init__(self) -> None:
        """Initialize StrategyEngine subsystems."""
        self._registry = StrategyRegistry()
        self._validator = StrategyValidator()
        self._pipeline = StrategyPipeline()
        self._exporter = StrategyExporter()

    @property
    def registry(self) -> StrategyRegistry:
        """Access StrategyRegistry."""
        return self._registry

    @property
    def validator(self) -> StrategyValidator:
        """Access StrategyValidator."""
        return self._validator

    @property
    def pipeline(self) -> StrategyPipeline:
        """Access StrategyPipeline."""
        return self._pipeline

    @property
    def exporter(self) -> StrategyExporter:
        """Access StrategyExporter."""
        return self._exporter

    def register_strategy(
        self, strategy_cls: Type[BaseStrategy], overwrite: bool = True
    ) -> None:
        """Register a strategy class into the central registry."""
        self._registry.register(strategy_cls, overwrite=overwrite)

    def enable_strategy(self, name: str) -> None:
        """Enable execution of a strategy by name."""
        self._registry.enable(name)

    def disable_strategy(self, name: str) -> None:
        """Disable execution of a strategy by name."""
        self._registry.disable(name)

    def execute_strategy(
        self,
        strategy_or_name: Any,
        data: pd.DataFrame,
        timeframe: str = "1h",
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Execute a strategy instance or registered strategy by name.

        Args:
            strategy_or_name: BaseStrategy instance or string strategy name.
            data: Input market DataFrame.
            timeframe: Active timeframe identifier.
            params: Optional hyperparameter overrides.

        Returns:
            pandas.DataFrame containing generated signal outputs.
        """
        if isinstance(strategy_or_name, str):
            if not self._registry.is_enabled(strategy_or_name):
                raise RuntimeError(f"Strategy '{strategy_or_name}' is disabled.")
            strat_cls = self._registry.get(strategy_or_name)
            strategy = strat_cls(params=params)
        else:
            strategy = strategy_or_name
            if params:
                strategy.initialize(params)

        output_df, report = self._pipeline.run(strategy, data, timeframe=timeframe)
        if not report.is_valid:
            raise ValueError(
                f"Strategy '{strategy.metadata().name}' execution failed validation: {report.errors}"
            )
        return output_df

    def new_builder(self, name: str = "CustomStrategy") -> StrategyBuilder:
        """Acquire a new StrategyBuilder instance."""
        return StrategyBuilder(name=name)

    def get_optimizer(self, algorithm_name: str) -> BaseOptimizer:
        """Acquire a strategy hyperparameter optimizer by algorithm name."""
        return OptimizerFactory.get_optimizer(algorithm_name)
