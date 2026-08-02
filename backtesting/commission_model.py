"""
QuantLab Institutional Commission Models.

Provides pluggable commission models for transaction fee calculation during simulation.
Supports Fixed, Percentage, Per Lot, Broker Specific, and Custom callback commission models.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class BaseCommissionModel(ABC):
    """Abstract Base Class for all commission models."""

    @abstractmethod
    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Calculate commission fee for a trade.

        Args:
            quantity: Trade quantity / volume.
            price: Executed trade price.
            side: 'BUY' or 'SELL'.
            notional: Optional notional trade value (quantity * price).

        Returns:
            Calculated commission cost in account base currency.
        """
        pass


class FixedCommissionModel(BaseCommissionModel):
    """Fixed fee per trade order regardless of size."""

    def __init__(self, fee_per_order: float = 1.0) -> None:
        """Initialize fixed commission model.

        Args:
            fee_per_order: Fixed currency fee per executed order.
        """
        if fee_per_order < 0:
            raise ValueError("fee_per_order must be non-negative.")
        self._fee = float(fee_per_order)

    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Return fixed fee per order."""
        return self._fee


class PercentageCommissionModel(BaseCommissionModel):
    """Percentage of notional order value commission model (e.g., 0.001 = 0.1%)."""

    def __init__(self, percentage: float = 0.001) -> None:
        """Initialize percentage commission model.

        Args:
            percentage: Decimal commission rate (e.g. 0.0005 for 0.05%).
        """
        if percentage < 0:
            raise ValueError("percentage must be non-negative.")
        self._percentage = float(percentage)

    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Calculate percentage of trade notional value."""
        val = notional if notional is not None else (quantity * price)
        return abs(val) * self._percentage


class PerLotCommissionModel(BaseCommissionModel):
    """Commission model based on cost per lot / unit (e.g., $7 per 100,000 unit lot)."""

    def __init__(self, cost_per_lot: float = 7.0, lot_size: float = 100000.0) -> None:
        """Initialize per lot commission model.

        Args:
            cost_per_lot: Fee charged per standard lot.
            lot_size: Number of base units per standard lot.
        """
        if cost_per_lot < 0 or lot_size <= 0:
            raise ValueError("cost_per_lot must be >= 0 and lot_size must be > 0.")
        self._cost_per_lot = float(cost_per_lot)
        self._lot_size = float(lot_size)

    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Calculate commission based on volume lot count."""
        lots = abs(quantity) / self._lot_size
        return lots * self._cost_per_lot


class BrokerSpecificCommissionModel(BaseCommissionModel):
    """Pre-configured institutional broker tier commission models."""

    PROFILES: Dict[str, Dict[str, Any]] = {
        "interactive_brokers_tiered": {
            "type": "tiered",
            "per_share": 0.0035,
            "min_per_order": 0.35,
            "max_pct": 0.01,
        },
        "forex_ecn_raw": {
            "type": "per_lot",
            "cost_per_lot": 6.0,
            "lot_size": 100000.0,
        },
        "crypto_binance_taker": {
            "type": "percentage",
            "percentage": 0.0010,
        },
        "crypto_binance_maker": {
            "type": "percentage",
            "percentage": 0.0005,
        },
        "futures_flat": {
            "type": "fixed",
            "fee_per_order": 2.25,
        },
    }

    def __init__(self, broker_profile: str = "forex_ecn_raw") -> None:
        """Initialize broker specific profile model.

        Args:
            broker_profile: Key of pre-configured broker profile.
        """
        if broker_profile not in self.PROFILES:
            raise ValueError(
                f"Unknown broker_profile '{broker_profile}'. "
                f"Available options: {list(self.PROFILES.keys())}"
            )
        self._profile_name = broker_profile
        self._profile = self.PROFILES[broker_profile]

    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Calculate commission based on selected broker profile."""
        p_type = self._profile["type"]
        val = notional if notional is not None else (quantity * price)
        abs_qty = abs(quantity)

        if p_type == "tiered":
            fee = abs_qty * self._profile["per_share"]
            fee = max(fee, self._profile["min_per_order"])
            max_fee = abs(val) * self._profile["max_pct"]
            return min(fee, max_fee)
        elif p_type == "per_lot":
            lots = abs_qty / self._profile["lot_size"]
            return lots * self._profile["cost_per_lot"]
        elif p_type == "percentage":
            return abs(val) * self._profile["percentage"]
        elif p_type == "fixed":
            return float(self._profile["fee_per_order"])
        else:
            return 0.0


class CustomCommissionModel(BaseCommissionModel):
    """Custom commission model accepting a user-defined function or lambda."""

    def __init__(self, fn: Callable[[float, float, str, Optional[float]], float]) -> None:
        """Initialize custom commission model with callable.

        Args:
            fn: Callable accepting (quantity, price, side, notional) returning float fee.
        """
        if not callable(fn):
            raise TypeError("fn must be a callable object.")
        self._fn = fn

    def calculate(
        self, quantity: float, price: float, side: str = "BUY", notional: Optional[float] = None
    ) -> float:
        """Delegate calculation to custom function."""
        return float(self._fn(quantity, price, side, notional))


class CommissionModelFactory:
    """Factory to instantiate commission models from name or configuration dict."""

    @staticmethod
    def create(model_type: str, **kwargs) -> BaseCommissionModel:
        """Create commission model instance.

        Args:
            model_type: Type identifier ('fixed', 'percentage', 'per_lot', 'broker', 'custom').
            kwargs: Parameters forwarded to constructor.

        Returns:
            Instance of BaseCommissionModel.
        """
        m_type = model_type.lower().strip()
        if m_type == "fixed":
            return FixedCommissionModel(**kwargs)
        elif m_type in ("percentage", "percent", "pct"):
            return PercentageCommissionModel(**kwargs)
        elif m_type in ("per_lot", "perlot", "lot"):
            return PerLotCommissionModel(**kwargs)
        elif m_type in ("broker", "broker_specific"):
            return BrokerSpecificCommissionModel(**kwargs)
        elif m_type == "custom":
            return CustomCommissionModel(**kwargs)
        else:
            raise ValueError(f"Unknown commission model type '{model_type}'.")
