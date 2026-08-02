"""
QuantLab Modular Strategy Builder.

Enables fluid programmatic construction of quantitative strategies by combining
filters, entry conditions, confirmations, risk rules, and exit rules into a single
executable BaseStrategy class instance.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.strategy_metadata import StrategyMetadata
from strategies.composition import (
    BaseCondition,
    BaseFilter,
    BaseConfirmation,
    BaseExitRule,
    BaseRiskRule,
)


class ComposedStrategy(BaseStrategy):
    """Concrete BaseStrategy generated dynamically by StrategyBuilder."""

    def __init__(
        self,
        builder_name: str,
        filters: List[BaseFilter],
        conditions: List[BaseCondition],
        confirmations: List[BaseConfirmation],
        exit_rules: List[BaseExitRule],
        risk_rule: Optional[BaseRiskRule] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._builder_name = builder_name
        self._filters = filters
        self._conditions = conditions
        self._confirmations = confirmations
        self._exit_rules = exit_rules
        self._risk_rule = risk_rule
        super().__init__(params=params)

    @classmethod
    def metadata(cls) -> StrategyMetadata:
        return StrategyMetadata(
            name="ComposedStrategy",
            category="Custom",
            description="Dynamically composed strategy created via StrategyBuilder.",
        )

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        raw_signal = pd.Series(0, index=data.index)

        # 1. Evaluate Entry Conditions
        if self._conditions:
            entry_mask = self._conditions[0].evaluate(data)
            for cond in self._conditions[1:]:
                entry_mask = entry_mask & cond.evaluate(data)
            raw_signal[entry_mask] = 1

        # 2. Apply Filters
        for flt in self._filters:
            filter_mask = flt.filter(data)
            raw_signal[~filter_mask] = 0

        # 3. Apply Confirmations
        for conf in self._confirmations:
            raw_signal = conf.confirm(data, raw_signal)

        return pd.DataFrame({"signal": raw_signal}, index=data.index)


class StrategyBuilder:
    """Fluent Builder for constructing modular strategies."""

    def __init__(self, name: str = "CustomStrategy") -> None:
        self._name = name
        self._filters: List[BaseFilter] = []
        self._conditions: List[BaseCondition] = []
        self._confirmations: List[BaseConfirmation] = []
        self._exit_rules: List[BaseExitRule] = []
        self._risk_rule: Optional[BaseRiskRule] = None

    def add_filter(self, market_filter: BaseFilter) -> "StrategyBuilder":
        self._filters.append(market_filter)
        return self

    def add_entry_condition(self, condition: BaseCondition) -> "StrategyBuilder":
        self._conditions.append(condition)
        return self

    def add_confirmation(self, confirmation: BaseConfirmation) -> "StrategyBuilder":
        self._confirmations.append(confirmation)
        return self

    def add_exit_rule(self, exit_rule: BaseExitRule) -> "StrategyBuilder":
        self._exit_rules.append(exit_rule)
        return self

    def set_risk_rule(self, risk_rule: BaseRiskRule) -> "StrategyBuilder":
        self._risk_rule = risk_rule
        return self

    def build(self) -> BaseStrategy:
        """Construct and return executable ComposedStrategy instance."""
        return ComposedStrategy(
            builder_name=self._name,
            filters=self._filters,
            conditions=self._conditions,
            confirmations=self._confirmations,
            exit_rules=self._exit_rules,
            risk_rule=self._risk_rule,
        )
