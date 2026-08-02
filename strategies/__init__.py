"""QuantLab Strategy Framework Package."""

from strategies.strategy_metadata import StrategyMetadata
from strategies.base_strategy import BaseStrategy
from strategies.strategy_registry import (
    StrategyRegistry,
    StrategyAlreadyRegisteredError,
    StrategyNotFoundError,
)
from strategies.strategy_validator import StrategyValidator, StrategyValidationReport
from strategies.strategy_builder import StrategyBuilder, ComposedStrategy
from strategies.strategy_optimizer import (
    BaseOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    GeneticOptimizer,
    OptimizerFactory,
    OptimizationResult,
)
from strategies.strategy_exporter import StrategyExporter
from strategies.strategy_pipeline import StrategyPipeline
from strategies.strategy_engine import StrategyEngine
from strategies.composition import (
    BaseCondition,
    PriceThresholdCondition,
    IndicatorCrossCondition,
    CompositeCondition,
    BaseFilter,
    TrendFilter,
    VolatilityFilter,
    BaseConfirmation,
    MultiIndicatorConfirmation,
    BaseExitRule,
    StopLossExit,
    TakeProfitExit,
    TrailingStopExit,
    BaseRiskRule,
    FixedFractionRisk,
    ATRPositionSizing,
)

__all__ = [
    "StrategyMetadata",
    "BaseStrategy",
    "StrategyRegistry",
    "StrategyAlreadyRegisteredError",
    "StrategyNotFoundError",
    "StrategyValidator",
    "StrategyValidationReport",
    "StrategyBuilder",
    "ComposedStrategy",
    "BaseOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    "OptimizerFactory",
    "OptimizationResult",
    "StrategyExporter",
    "StrategyPipeline",
    "StrategyEngine",
    "BaseCondition",
    "PriceThresholdCondition",
    "IndicatorCrossCondition",
    "CompositeCondition",
    "BaseFilter",
    "TrendFilter",
    "VolatilityFilter",
    "BaseConfirmation",
    "MultiIndicatorConfirmation",
    "BaseExitRule",
    "StopLossExit",
    "TakeProfitExit",
    "TrailingStopExit",
    "BaseRiskRule",
    "FixedFractionRisk",
    "ATRPositionSizing",
]
