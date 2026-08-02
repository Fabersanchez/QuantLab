"""
QuantLab Institutional Order Manager and Order Data Structures.

Provides order lifecycle management, validation, pending order tracking,
and support for Market, Limit, Stop, and Stop-Limit order types with Time-in-Force policies.
"""

from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Dict, List, Optional, Any


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Order side (BUY or SELL)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class TimeInForce(str, Enum):
    """Time-In-Force order execution policies."""

    GTC = "GTC"  # Good 'Til Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Good for Day
    BAR_EXPIRY = "BAR_EXPIRY"  # Expires after N bars


@dataclass
class Order:
    """Dataclass representing a trading order in QuantLab."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    filled_quantity: float = 0.0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: Any = None
    created_bar_index: int = 0
    filled_at: Any = None
    expires_at: Any = None
    expires_bar_index: Optional[int] = None
    commission: float = 0.0
    avg_fill_price: float = 0.0
    rejection_reason: Optional[str] = None
    parent_position_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pips: Optional[float] = None

    @property
    def remaining_quantity(self) -> float:
        """Return unfilled order quantity."""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_active(self) -> bool:
        """Return True if order is pending or partially filled."""
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)


class OrderManager:
    """Manages order creation, lifecycle transitions, state tracking, and query operations."""

    def __init__(self) -> None:
        """Initialize OrderManager."""
        self._orders: Dict[str, Order] = {}
        self._counter: int = 0

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        created_at: Any = None,
        created_bar_index: int = 0,
        expires_bar_index: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pips: Optional[float] = None,
        parent_position_id: Optional[str] = None,
    ) -> Order:
        """Create and register a new order.

        Args:
            symbol: Asset symbol.
            side: OrderSide.BUY or OrderSide.SELL.
            order_type: OrderType (MARKET, LIMIT, STOP, STOP_LIMIT).
            quantity: Order volume / shares / contracts.
            limit_price: Mandatory for LIMIT and STOP_LIMIT orders.
            stop_price: Mandatory for STOP and STOP_LIMIT orders.
            time_in_force: TimeInForce policy.
            created_at: Timestamp of creation.
            created_bar_index: Bar index of creation.
            expires_bar_index: Expiration bar index if BAR_EXPIRY.
            stop_loss: Optional attached SL price.
            take_profit: Optional attached TP price.
            trailing_stop_pips: Optional attached trailing stop pips.
            parent_position_id: Associated position ID.

        Returns:
            Newly created Order instance.
        """
        if quantity <= 0:
            raise ValueError("Order quantity must be positive.")

        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and limit_price is None:
            raise ValueError(f"limit_price is required for order type '{order_type}'.")

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise ValueError(f"stop_price is required for order type '{order_type}'.")

        self._counter += 1
        order_id = f"ORD-{self._counter:06d}-{uuid.uuid4().hex[:6]}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            limit_price=float(limit_price) if limit_price is not None else None,
            stop_price=float(stop_price) if stop_price is not None else None,
            status=OrderStatus.PENDING,
            time_in_force=time_in_force,
            created_at=created_at,
            created_bar_index=created_bar_index,
            expires_bar_index=expires_bar_index,
            stop_loss=float(stop_loss) if stop_loss is not None else None,
            take_profit=float(take_profit) if take_profit is not None else None,
            trailing_stop_pips=float(trailing_stop_pips) if trailing_stop_pips is not None else None,
            parent_position_id=parent_position_id,
        )

        self._orders[order_id] = order
        return order

    def update_order_fill(
        self,
        order_id: str,
        fill_qty: float,
        fill_price: float,
        commission: float = 0.0,
        fill_time: Any = None,
    ) -> Order:
        """Record order fill execution (partial or complete).

        Args:
            order_id: Target order identifier.
            fill_qty: Executed fill quantity.
            fill_price: Execution price for this fill.
            commission: Commission incurred on this fill.
            fill_time: Timestamp of execution.

        Returns:
            Updated Order object.
        """
        order = self.get_order(order_id)
        if not order:
            raise KeyError(f"Order '{order_id}' not found.")

        if not order.is_active:
            raise RuntimeError(f"Cannot fill order '{order_id}' in state '{order.status}'.")

        fill_qty = min(fill_qty, order.remaining_quantity)
        total_prev_val = order.avg_fill_price * order.filled_quantity
        new_val = total_prev_val + (fill_price * fill_qty)

        order.filled_quantity += fill_qty
        order.avg_fill_price = new_val / order.filled_quantity if order.filled_quantity > 0 else 0.0
        order.commission += commission
        order.filled_at = fill_time

        if order.filled_quantity >= order.quantity - 1e-9:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        return order

    def cancel_order(self, order_id: str, reason: str = "Cancelled by user") -> Order:
        """Cancel a pending or partially filled order.

        Args:
            order_id: Target order ID.
            reason: Reason description for cancellation.

        Returns:
            Updated Order object.
        """
        order = self.get_order(order_id)
        if not order:
            raise KeyError(f"Order '{order_id}' not found.")

        if order.is_active:
            order.status = OrderStatus.CANCELLED
            order.rejection_reason = reason

        return order

    def reject_order(self, order_id: str, reason: str) -> Order:
        """Reject an order due to margin check or validation failure.

        Args:
            order_id: Target order ID.
            reason: Detailed rejection reason.

        Returns:
            Updated Order object.
        """
        order = self.get_order(order_id)
        if not order:
            raise KeyError(f"Order '{order_id}' not found.")

        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason
        return order

    def expire_order(self, order_id: str, reason: str = "TimeInForce Expiry") -> Order:
        """Mark an order as expired.

        Args:
            order_id: Target order ID.
            reason: Expiry detail.

        Returns:
            Updated Order object.
        """
        order = self.get_order(order_id)
        if not order:
            raise KeyError(f"Order '{order_id}' not found.")

        order.status = OrderStatus.EXPIRED
        order.rejection_reason = reason
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Fetch all active pending/partially-filled orders."""
        active = [o for o in self._orders.values() if o.is_active]
        if symbol:
            active = [o for o in active if o.symbol.upper() == symbol.upper()]
        return active

    def get_all_orders(self) -> List[Order]:
        """Fetch list of all orders."""
        return list(self._orders.values())

    def clear(self) -> None:
        """Reset internal order registry."""
        self._orders.clear()
        self._counter = 0
