"""
QuantLab Equity Curve Container and Timeseries Analytics.

Tracks step-by-step account equity, cash balance, margin usage, high-water mark,
drawdown series, underwater curves, and return series calculations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class EquityPoint:
    """Dataclass representing an individual point on the portfolio equity curve."""

    timestamp: Any
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    drawdown_amount: float
    drawdown_pct: float
    open_positions_count: int


class EquityCurve:
    """Institutional Equity Curve Tracker."""

    def __init__(self) -> None:
        """Initialize EquityCurve tracker."""
        self._points: List[EquityPoint] = []

    def add_point(
        self,
        timestamp: Any,
        balance: float,
        equity: float,
        margin_used: float = 0.0,
        free_margin: float = 0.0,
        drawdown_amount: float = 0.0,
        drawdown_pct: float = 0.0,
        open_positions_count: int = 0,
    ) -> EquityPoint:
        """Record a new equity curve snapshot point.

        Returns:
            EquityPoint instance added.
        """
        point = EquityPoint(
            timestamp=timestamp,
            balance=float(balance),
            equity=float(equity),
            margin_used=float(margin_used),
            free_margin=float(free_margin),
            drawdown_amount=float(drawdown_amount),
            drawdown_pct=float(drawdown_pct),
            open_positions_count=int(open_positions_count),
        )
        self._points.append(point)
        return point

    def to_dataframe(self) -> pd.DataFrame:
        """Convert equity curve history into a pandas DataFrame indexed by timestamp."""
        if not self._points:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "balance",
                    "equity",
                    "margin_used",
                    "free_margin",
                    "drawdown_amount",
                    "drawdown_pct",
                    "open_positions_count",
                ]
            )

        data = [
            {
                "timestamp": p.timestamp,
                "balance": p.balance,
                "equity": p.equity,
                "margin_used": p.margin_used,
                "free_margin": p.free_margin,
                "drawdown_amount": p.drawdown_amount,
                "drawdown_pct": p.drawdown_pct,
                "open_positions_count": p.open_positions_count,
            }
            for p in self._points
        ]
        df = pd.DataFrame(data)
        if "timestamp" in df.columns and df["timestamp"].notnull().any():
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            except Exception:
                pass
        return df

    def get_equity_series(self) -> pd.Series:
        """Return equity series as pandas Series."""
        df = self.to_dataframe()
        return df["equity"] if not df.empty and "equity" in df.columns else pd.Series(dtype=float)

    def get_returns_series(self) -> pd.Series:
        """Return percentage returns series (pct_change) of equity."""
        eq = self.get_equity_series()
        if eq.empty:
            return pd.Series(dtype=float)
        return eq.pct_change().fillna(0.0)

    def get_drawdown_series(self) -> pd.Series:
        """Return drawdown percentage series."""
        df = self.to_dataframe()
        return df["drawdown_pct"] if not df.empty and "drawdown_pct" in df.columns else pd.Series(dtype=float)

    def get_underwater_series(self) -> pd.Series:
        """Return underwater curve (negative drawdown percentage)."""
        dd = self.get_drawdown_series()
        return -dd.abs()

    def calculate_peak_equity(self) -> float:
        """Return maximum high water mark peak equity recorded."""
        if not self._points:
            return 0.0
        return max(p.equity for p in self._points)

    def calculate_max_drawdown(self) -> Tuple[float, float]:
        """Return max drawdown tuple (max_dd_amount, max_dd_pct)."""
        if not self._points:
            return (0.0, 0.0)
        max_amt = max(p.drawdown_amount for p in self._points)
        max_pct = max(p.drawdown_pct for p in self._points)
        return (max_amt, max_pct)

    def clear(self) -> None:
        """Clear recorded points."""
        self._points.clear()

    def __len__(self) -> int:
        return len(self._points)
