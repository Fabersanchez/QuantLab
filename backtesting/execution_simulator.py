"""
QuantLab Realistic Execution Simulator.

Simulates order execution logic against OHLC bar data while incorporating commission,
slippage, bid-ask spread, latency delays, liquidity volume participation limits,
partial fills, order rejections, and Time-in-Force expiration rules.
"""

from typing import Any, Dict, List, Optional, Tuple
from backtesting.commission_model import BaseCommissionModel, FixedCommissionModel
from backtesting.slippage_model import BaseSlippageModel, FixedSlippageModel
from backtesting.spread_model import BaseSpreadModel, FixedSpreadModel
from backtesting.latency_model import BaseLatencyModel, ExecutionDelayModel
from backtesting.order_manager import Order, OrderManager, OrderSide, OrderStatus, OrderType, TimeInForce
from backtesting.position_manager import Position, PositionManager, PositionSide
from backtesting.portfolio_manager import PortfolioManager


class ExecutionSimulator:
    """Institutional Order Execution Engine."""

    def __init__(
        self,
        commission_model: Optional[BaseCommissionModel] = None,
        slippage_model: Optional[BaseSlippageModel] = None,
        spread_model: Optional[BaseSpreadModel] = None,
        latency_model: Optional[BaseLatencyModel] = None,
        max_volume_participation: float = 0.1,  # Max 10% of bar volume per fill
        point_value: float = 0.0001,
    ) -> None:
        """Initialize ExecutionSimulator.

        Args:
            commission_model: Model for transaction fee calculations.
            slippage_model: Model for price slippage adjustments.
            spread_model: Model for bid-ask spread.
            latency_model: Model for latency offsets.
            max_volume_participation: Maximum fraction of bar volume allowed for partial fills.
            point_value: Size of 1 pip for instrument.
        """
        self.commission_model = commission_model or FixedCommissionModel(0.0)
        self.slippage_model = slippage_model or FixedSlippageModel(0.0)
        self.spread_model = spread_model or FixedSpreadModel(0.0)
        self.latency_model = latency_model or ExecutionDelayModel(0)
        self.max_volume_participation = max(0.001, float(max_volume_participation))
        self.point_value = float(point_value)

    def process_order_queues(
        self,
        bar_data: Dict[str, Any],
        bar_index: int,
        order_manager: OrderManager,
        position_manager: PositionManager,
        portfolio_manager: PortfolioManager,
    ) -> Tuple[List[Order], List[Position]]:
        """Evaluate pending orders and process execution against current bar data.

        Args:
            bar_data: Dict containing 'symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp'.
            bar_index: Current sequential bar index.
            order_manager: System OrderManager instance.
            position_manager: System PositionManager instance.
            portfolio_manager: System PortfolioManager instance.

        Returns:
            Tuple of (executed_orders_list, affected_positions_list).
        """
        symbol = bar_data.get("symbol", "GENERIC").upper()
        open_p = float(bar_data.get("open", bar_data.get("close", 0.0)))
        high_p = float(bar_data.get("high", open_p))
        low_p = float(bar_data.get("low", open_p))
        close_p = float(bar_data.get("close", open_p))
        volume = float(bar_data.get("volume", 1000000.0))
        timestamp = bar_data.get("timestamp")

        executed_orders: List[Order] = []
        affected_positions: List[Position] = []

        pending_orders = order_manager.get_open_orders(symbol=symbol)
        for order in pending_orders:
            # 1. Check Latency Delay
            bar_delay, _ = self.latency_model.calculate_delay(timestamp, bar_index)
            if bar_index < order.created_bar_index + bar_delay:
                continue  # Order delayed by latency model

            # 2. Check Expiration
            if order.time_in_force == TimeInForce.BAR_EXPIRY:
                if order.expires_bar_index is not None and bar_index >= order.expires_bar_index:
                    order_manager.expire_order(order.order_id, "Bar expiry reached")
                    continue

            # 3. Calculate Liquidity Limit for Partial Fill
            max_fillable_qty = volume * self.max_volume_participation
            fill_qty = min(order.remaining_quantity, max_fillable_qty)

            if fill_qty <= 0:
                continue

            # Handle IOC / FOK Policies if volume insufficient
            if order.time_in_force == TimeInForce.FOK and fill_qty < order.remaining_quantity:
                order_manager.cancel_order(order.order_id, "FOK failed volume check")
                continue

            # 4. Check Order Execution Conditions & Trigger Prices
            fill_price, triggered = self._evaluate_order_trigger(
                order, open_p, high_p, low_p, close_p, timestamp, bar_data
            )

            if not triggered or fill_price is None:
                if order.time_in_force == TimeInForce.IOC and bar_index > order.created_bar_index:
                    order_manager.cancel_order(order.order_id, "IOC expired unfilled")
                continue

            # 5. Calculate Bid-Ask Spread & Slippage
            bid, ask = self.spread_model.get_bid_ask(timestamp, fill_price, bar_data)
            base_price = ask if order.side == OrderSide.BUY else bid

            exec_price = self.slippage_model.calculate_execution_price(
                base_price, fill_qty, order.side.value, bar_data
            )

            # 6. Calculate Commission
            notional = fill_qty * exec_price
            commission = self.commission_model.calculate(fill_qty, exec_price, order.side.value, notional)

            # 7. Check Margin Availability in PortfolioManager
            required_margin = portfolio_manager.calculate_required_margin(notional)
            if not portfolio_manager.check_margin_availability(required_margin):
                order_manager.reject_order(order.order_id, "Insufficient Free Margin")
                continue

            # 8. Execute Order & Update OrderManager
            slippage_cost = self.slippage_model.calculate_slippage_amount(
                base_price, fill_qty, order.side.value, bar_data
            ) * fill_qty

            order_manager.update_order_fill(
                order.order_id, fill_qty, exec_price, commission, timestamp
            )
            executed_orders.append(order)

            # 9. Create or Update Position in PositionManager
            pos_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT

            # Check if matching open position exists for scaling in
            existing_positions = [
                p for p in position_manager.get_open_positions(symbol)
                if p.side == pos_side
            ]

            if existing_positions:
                pos = existing_positions[0]
                pos.scale_in(fill_qty, exec_price, commission, slippage_cost, timestamp)
                affected_positions.append(pos)
            else:
                pos = position_manager.open_position(
                    symbol=symbol,
                    side=pos_side,
                    quantity=fill_qty,
                    entry_price=exec_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    trailing_stop_pips=order.trailing_stop_pips,
                    commission=commission,
                    slippage=slippage_cost,
                    opened_at=timestamp,
                )
                affected_positions.append(pos)

        return (executed_orders, affected_positions)

    def _evaluate_order_trigger(
        self,
        order: Order,
        open_p: float,
        high_p: float,
        low_p: float,
        close_p: float,
        timestamp: Any,
        bar_data: Dict[str, Any],
    ) -> Tuple[Optional[float], bool]:
        """Evaluate if price trigger conditions for Market, Limit, Stop, Stop-Limit are met.

        Returns (fill_price, is_triggered).
        """
        o_type = order.order_type

        if o_type == OrderType.MARKET:
            return (open_p, True)

        elif o_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if low_p <= order.limit_price:
                    # Fill at limit price or gap open price if opened better
                    fill = min(open_p, order.limit_price)
                    return (fill, True)
            else:  # SELL LIMIT
                if high_p >= order.limit_price:
                    fill = max(open_p, order.limit_price)
                    return (fill, True)

        elif o_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                if high_p >= order.stop_price:
                    fill = max(open_p, order.stop_price)
                    return (fill, True)
            else:  # SELL STOP
                if low_p <= order.stop_price:
                    fill = min(open_p, order.stop_price)
                    return (fill, True)

        elif o_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                if high_p >= order.stop_price and low_p <= order.limit_price:
                    return (order.limit_price, True)
            else:
                if low_p <= order.stop_price and high_p >= order.limit_price:
                    return (order.limit_price, True)

        return (None, False)
