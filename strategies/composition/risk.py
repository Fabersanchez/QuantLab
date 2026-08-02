"""Strategy Risk and Position Sizing Rules."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseRiskRule(ABC):
    """Abstract Base Class for risk management and position sizing rules."""

    @abstractmethod
    def calculate_position_size(
        self, equity: float, entry_price: float, stop_loss_price: float
    ) -> float:
        """Calculate trade position size in units or contracts."""
        pass


class FixedFractionRisk(BaseRiskRule):
    """Fixed Fractional Risk sizing rule (risk X% of equity per trade)."""

    def __init__(self, risk_fraction: float = 0.01) -> None:
        self.risk_fraction = risk_fraction

    def calculate_position_size(
        self, equity: float, entry_price: float, stop_loss_price: float
    ) -> float:
        risk_amount = equity * self.risk_fraction
        per_unit_risk = abs(entry_price - stop_loss_price)
        if per_unit_risk <= 0:
            return 0.0
        return risk_amount / per_unit_risk


class ATRPositionSizing(BaseRiskRule):
    """ATR Volatility-adjusted position sizing rule."""

    def __init__(self, risk_fraction: float = 0.01, atr_multiplier: float = 2.0) -> None:
        self.risk_fraction = risk_fraction
        self.atr_multiplier = atr_multiplier

    def calculate_position_size(
        self, equity: float, entry_price: float, atr_value: float
    ) -> float:
        risk_amount = equity * self.risk_fraction
        per_unit_risk = atr_value * self.atr_multiplier
        if per_unit_risk <= 0:
            return 0.0
        return risk_amount / per_unit_risk
