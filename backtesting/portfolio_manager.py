"""
QuantLab Institutional Portfolio Manager.

Tracks account balance, unrealized/realized equity, margin usage, leverage,
exposure, peak equity high-water mark, drawdown, and margin call / stop out liquidation triggers.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
from backtesting.position_manager import Position, PositionSide


@dataclass
class PortfolioState:
    """Dataclass representing instantaneous snapshot of portfolio account state."""

    timestamp: Any
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_level_pct: float
    unrealized_pnl: float
    realized_pnl: float
    notional_exposure: float
    leverage: float
    drawdown_amount: float
    drawdown_pct: float


class PortfolioManager:
    """Institutional Portfolio and Account Risk Manager."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        leverage: float = 100.0,
        margin_call_pct: float = 100.0,
        stop_out_pct: float = 50.0,
    ) -> None:
        """Initialize PortfolioManager.

        Args:
            initial_capital: Account starting balance in cash currency.
            leverage: Maximum account leverage ratio (e.g. 100.0 for 100:1).
            margin_call_pct: Margin level percentage threshold for margin call warning.
            stop_out_pct: Margin level percentage threshold for forced liquidation stop out.
        """
        if initial_capital <= 0 or leverage <= 0:
            raise ValueError("initial_capital and leverage must be positive.")

        self._initial_capital = float(initial_capital)
        self._leverage = float(leverage)
        self._margin_call_pct = float(margin_call_pct)
        self._stop_out_pct = float(stop_out_pct)

        self._balance: float = self._initial_capital
        self._cum_realized_pnl: float = 0.0
        self._peak_equity: float = self._initial_capital
        self._current_state: Optional[PortfolioState] = None

    @property
    def initial_capital(self) -> float:
        """Return initial starting capital."""
        return self._initial_capital

    @property
    def balance(self) -> float:
        """Return current cash balance."""
        return self._balance

    @property
    def leverage(self) -> float:
        """Return configured leverage ratio."""
        return self._leverage

    @property
    def peak_equity(self) -> float:
        """Return historical high water mark peak equity."""
        return self._peak_equity

    def calculate_required_margin(self, notional_value: float) -> float:
        """Calculate required margin for a position or order.

        Args:
            notional_value: Position value (quantity * price).

        Returns:
            Required margin currency amount.
        """
        return abs(notional_value) / self._leverage

    def check_margin_availability(self, required_margin: float) -> bool:
        """Check if current free margin is sufficient for new position margin requirement."""
        if not self._current_state:
            return required_margin <= self._balance
        return required_margin <= self._current_state.free_margin

    def process_realized_pnl(self, pnl: float) -> float:
        """Update cash balance with trade realized PnL.

        Args:
            pnl: Realized profit/loss.

        Returns:
            New updated balance.
        """
        self._balance += pnl
        self._cum_realized_pnl += pnl
        return self._balance

    def update_state(self, timestamp: Any, open_positions: List[Position]) -> PortfolioState:
        """Calculate and return snapshot of current portfolio state.

        Args:
            timestamp: Current simulation bar/tick timestamp.
            open_positions: List of currently open active positions.

        Returns:
            PortfolioState dataclass.
        """
        unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)
        equity = self._balance + unrealized_pnl
        self._peak_equity = max(self._peak_equity, equity)

        notional_exposure = sum(p.quantity * p.current_price for p in open_positions)
        margin_used = sum(self.calculate_required_margin(p.quantity * p.current_price) for p in open_positions)
        free_margin = max(0.0, equity - margin_used)

        if margin_used > 0:
            margin_level_pct = (equity / margin_used) * 100.0
        else:
            margin_level_pct = 999999.0  # Infinite margin level when no open margin used

        drawdown_amount = max(0.0, self._peak_equity - equity)
        drawdown_pct = (drawdown_amount / self._peak_equity) * 100.0 if self._peak_equity > 0 else 0.0

        state = PortfolioState(
            timestamp=timestamp,
            balance=self._balance,
            equity=equity,
            margin_used=margin_used,
            free_margin=free_margin,
            margin_level_pct=margin_level_pct,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._cum_realized_pnl,
            notional_exposure=notional_exposure,
            leverage=self._leverage,
            drawdown_amount=drawdown_amount,
            drawdown_pct=drawdown_pct,
        )

        self._current_state = state
        return state

    def check_stop_out(self, state: Optional[PortfolioState] = None) -> bool:
        """Check if margin level has breached forced stop out threshold.

        Returns:
            True if forced liquidation stop out is triggered.
        """
        st = state or self._current_state
        if not st or st.margin_used <= 0:
            return False
        return st.margin_level_pct <= self._stop_out_pct

    def check_margin_call(self, state: Optional[PortfolioState] = None) -> bool:
        """Check if margin level has breached warning margin call threshold."""
        st = state or self._current_state
        if not st or st.margin_used <= 0:
            return False
        return st.margin_level_pct <= self._margin_call_pct

    def reset() -> None:
        """Reset portfolio state to initial capital."""
        self._balance = self._initial_capital
        self._cum_realized_pnl = 0.0
        self._peak_equity = self._initial_capital
        self._current_state = None
