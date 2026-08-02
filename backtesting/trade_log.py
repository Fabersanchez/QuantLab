"""
QuantLab Granular Trade Log Container.

Records comprehensive audit logs for all executed trade transactions including entry/exit times,
prices, reasons, attached indicators/features, commissions, slippage, MAE/MFE metrics, and export utilities.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class TradeRecord:
    """Dataclass representing an audited, closed trade transaction."""

    trade_id: str
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_time: Any
    entry_price: float
    exit_time: Any
    exit_price: float
    holding_duration_seconds: float = 0.0
    holding_bars: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exit_reason: str = "UNKNOWN"
    indicators_at_entry: Dict[str, Any] = field(default_factory=dict)
    features_at_entry: Dict[str, Any] = field(default_factory=dict)
    mae: float = 0.0  # Maximum Adverse Excursion (maximum loss during holding)
    mfe: float = 0.0  # Maximum Favorable Excursion (maximum unrealized profit during holding)

    @property
    def is_win(self) -> bool:
        """Return True if net PnL is positive."""
        return self.net_pnl > 0.0


class TradeLog:
    """Institutional Trade Log Recorder and Filter."""

    def __init__(self) -> None:
        """Initialize TradeLog."""
        self._trades: List[TradeRecord] = []
        self._counter: int = 0

    def record_trade(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_time: Any,
        entry_price: float,
        exit_time: Any,
        exit_price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        exit_reason: str = "SIGNAL",
        indicators_at_entry: Optional[Dict[str, Any]] = None,
        features_at_entry: Optional[Dict[str, Any]] = None,
        holding_bars: int = 0,
        mae: float = 0.0,
        mfe: float = 0.0,
    ) -> TradeRecord:
        """Record and append a closed trade record.

        Returns:
            Recorded TradeRecord instance.
        """
        self._counter += 1
        trade_id = f"TRD-{self._counter:06d}"

        # Gross PnL
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity

        net_pnl = gross_pnl - (commission + slippage)
        notional_entry = entry_price * quantity
        pnl_pct = (net_pnl / notional_entry) * 100.0 if notional_entry > 0 else 0.0

        # Holding duration
        duration_sec = 0.0
        if entry_time and exit_time:
            try:
                t_in = pd.to_datetime(entry_time)
                t_out = pd.to_datetime(exit_time)
                duration_sec = (t_out - t_in).total_seconds()
            except Exception:
                duration_sec = 0.0

        record = TradeRecord(
            trade_id=trade_id,
            position_id=position_id,
            symbol=symbol,
            side=side_upper,
            quantity=float(quantity),
            entry_time=entry_time,
            entry_price=float(entry_price),
            exit_time=exit_time,
            exit_price=float(exit_price),
            holding_duration_seconds=duration_sec,
            holding_bars=holding_bars,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=float(commission),
            slippage=float(slippage),
            exit_reason=exit_reason,
            indicators_at_entry=indicators_at_entry or {},
            features_at_entry=features_at_entry or {},
            mae=float(mae),
            mfe=float(mfe),
        )

        self._trades.append(record)
        return record

    def get_all_trades(self) -> List[TradeRecord]:
        """Fetch all recorded trades."""
        return list(self._trades)

    def get_winning_trades(self) -> List[TradeRecord]:
        """Fetch all winning trades (net PnL > 0)."""
        return [t for t in self._trades if t.is_win]

    def get_losing_trades(self) -> List[TradeRecord]:
        """Fetch all losing trades (net PnL <= 0)."""
        return [t for t in self._trades if not t.is_win]

    def filter_by_symbol(self, symbol: str) -> List[TradeRecord]:
        """Filter trade log by asset symbol."""
        return [t for t in self._trades if t.symbol.upper() == symbol.upper()]

    def filter_by_side(self, side: str) -> List[TradeRecord]:
        """Filter trade log by direction ('LONG' or 'SHORT')."""
        return [t for t in self._trades if t.side.upper() == side.upper()]

    def to_dataframe(self) -> pd.DataFrame:
        """Export trade log to pandas DataFrame."""
        if not self._trades:
            return pd.DataFrame(
                columns=[
                    "trade_id",
                    "position_id",
                    "symbol",
                    "side",
                    "quantity",
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "gross_pnl",
                    "net_pnl",
                    "pnl_pct",
                    "commission",
                    "slippage",
                    "exit_reason",
                    "holding_bars",
                    "holding_duration_seconds",
                    "mae",
                    "mfe",
                ]
            )

        data = []
        for t in self._trades:
            data.append(
                {
                    "trade_id": t.trade_id,
                    "position_id": t.position_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "pnl_pct": t.pnl_pct,
                    "commission": t.commission,
                    "slippage": t.slippage,
                    "exit_reason": t.exit_reason,
                    "holding_bars": t.holding_bars,
                    "holding_duration_seconds": t.holding_duration_seconds,
                    "mae": t.mae,
                    "mfe": t.mfe,
                }
            )
        return pd.DataFrame(data)

    def export_csv(self, filepath: str) -> None:
        """Export trade log to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def export_json(self, filepath: str) -> None:
        """Export trade log to JSON file."""
        df = self.to_dataframe()
        df.to_json(filepath, orient="records", date_format="iso", indent=2)

    def clear(self) -> None:
        """Clear trade log records."""
        self._trades.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._trades)
