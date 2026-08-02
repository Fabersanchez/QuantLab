"""Strategy Composition Package."""

from strategies.composition.conditions import (
    BaseCondition,
    PriceThresholdCondition,
    IndicatorCrossCondition,
    CompositeCondition,
)
from strategies.composition.filters import (
    BaseFilter,
    TrendFilter,
    VolatilityFilter,
)
from strategies.composition.confirmations import (
    BaseConfirmation,
    MultiIndicatorConfirmation,
)
from strategies.composition.exits import (
    BaseExitRule,
    StopLossExit,
    TakeProfitExit,
    TrailingStopExit,
)
from strategies.composition.risk import (
    BaseRiskRule,
    FixedFractionRisk,
    ATRPositionSizing,
)

__all__ = [
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
