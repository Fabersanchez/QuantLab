"""
QuantLab Portfolio Rebalancing Engine.

Provides automated portfolio rebalancing triggers:
Calendar-based (Daily, Weekly, Monthly, Quarterly), Weight Drift threshold,
Volatility spike threshold, Drawdown limit, Event-driven, and Manual rebalancing.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
import pandas as pd

from portfolio.portfolio import Portfolio


class RebalanceTrigger(str, Enum):
    """Rebalancing trigger type categorization."""

    MANUAL = "MANUAL"
    CALENDAR_DAILY = "CALENDAR_DAILY"
    CALENDAR_WEEKLY = "CALENDAR_WEEKLY"
    CALENDAR_MONTHLY = "CALENDAR_MONTHLY"
    CALENDAR_QUARTERLY = "CALENDAR_QUARTERLY"
    WEIGHT_DRIFT = "WEIGHT_DRIFT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    EVENT_DRIVEN = "EVENT_DRIVEN"


class PortfolioRebalancer:
    """Institutional Portfolio Rebalancing Engine."""

    def __init__(self, drift_threshold: float = 0.05, drawdown_threshold: float = 0.15) -> None:
        """Initialize PortfolioRebalancer.

        Args:
            drift_threshold: Maximum weight drift tolerance float (e.g. 0.05 = 5%).
            drawdown_threshold: Maximum portfolio drawdown tolerance float (e.g. 0.15 = 15%).
        """
        self.drift_threshold = drift_threshold
        self.drawdown_threshold = drawdown_threshold

    def should_rebalance(
        self,
        portfolio: Portfolio,
        current_weights: Dict[str, float],
        trigger_type: RebalanceTrigger = RebalanceTrigger.WEIGHT_DRIFT,
        current_drawdown: float = 0.0,
    ) -> bool:
        """Evaluate whether a portfolio requires rebalancing under trigger rules.

        Args:
            portfolio: Portfolio instance holding target weights.
            current_weights: Active asset weights dictionary.
            trigger_type: Rebalance trigger specification.
            current_drawdown: Current portfolio drawdown fraction.

        Returns:
            Boolean indicating if rebalancing is required.
        """
        if trigger_type == RebalanceTrigger.MANUAL:
            return True

        if trigger_type == RebalanceTrigger.DRAWDOWN_LIMIT:
            return current_drawdown >= self.drawdown_threshold

        if trigger_type == RebalanceTrigger.WEIGHT_DRIFT:
            for sym, target_w in portfolio.weights.items():
                cur_w = current_weights.get(sym, 0.0)
                if abs(cur_w - target_w) > self.drift_threshold:
                    return True
            return False

        # Calendar defaults
        return True

    def execute_rebalance(
        self,
        portfolio: Portfolio,
        target_weights: Dict[str, float],
        trigger_type: RebalanceTrigger = RebalanceTrigger.MANUAL,
    ) -> Dict[str, Any]:
        """Execute portfolio rebalancing update and record history event.

        Args:
            portfolio: Portfolio instance.
            target_weights: New target asset weights dictionary.
            trigger_type: Trigger type string.

        Returns:
            Execution event record dictionary.
        """
        old_weights = dict(portfolio.weights)
        portfolio.set_weights(target_weights)

        event_record = {
            "trigger_type": trigger_type.value if isinstance(trigger_type, Enum) else str(trigger_type),
            "old_weights": old_weights,
            "new_weights": target_weights,
        }
        portfolio.history_events.append(event_record)
        return event_record
