"""
QuantLab Simulation Replay & Step-by-Step Audit Engine.

Allows stepping through a completed backtest bar-by-bar or trade-by-trade to inspect open orders,
active positions, portfolio balance, equity, and indicator signals at any timestamp in history.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ReplaySnapshot:
    """Dataclass representing instantaneous backtest state snapshot for replay audit."""

    step_index: int
    timestamp: Any
    bar_data: Dict[str, Any]
    indicator_values: Dict[str, Any] = field(default_factory=dict)
    open_orders_count: int = 0
    open_positions_count: int = 0
    closed_positions_count: int = 0
    balance: float = 0.0
    equity: float = 0.0
    drawdown_pct: float = 0.0
    signal: int = 0


class BacktestReplay:
    """Institutional Replay and Audit Engine."""

    def __init__(self, snapshots: Optional[List[ReplaySnapshot]] = None) -> None:
        """Initialize BacktestReplay.

        Args:
            snapshots: Optional initial list of ReplaySnapshot objects.
        """
        self._snapshots: List[ReplaySnapshot] = snapshots or []
        self._current_index: int = 0

    def add_snapshot(self, snapshot: ReplaySnapshot) -> None:
        """Append snapshot to replay timeline."""
        self._snapshots.append(snapshot)

    @property
    def total_steps(self) -> int:
        """Return total steps/bars in replay dataset."""
        return len(self._snapshots)

    @property
    def current_step(self) -> int:
        """Return current active step index."""
        return self._current_index

    def get_current_snapshot(self) -> Optional[ReplaySnapshot]:
        """Fetch active snapshot at current index."""
        if 0 <= self._current_index < len(self._snapshots):
            return self._snapshots[self._current_index]
        return None

    def step_forward(self) -> Optional[ReplaySnapshot]:
        """Step one bar forward in time.

        Returns:
            New active ReplaySnapshot.
        """
        if self._current_index < len(self._snapshots) - 1:
            self._current_index += 1
            return self.get_current_snapshot()
        return self.get_current_snapshot()

    def step_backward(self) -> Optional[ReplaySnapshot]:
        """Step one bar backward in time.

        Returns:
            New active ReplaySnapshot.
        """
        if self._current_index > 0:
            self._current_index -= 1
            return self.get_current_snapshot()
        return self.get_current_snapshot()

    def jump_to_step(self, step_index: int) -> Optional[ReplaySnapshot]:
        """Jump directly to specified bar step index."""
        if 0 <= step_index < len(self._snapshots):
            self._current_index = step_index
            return self.get_current_snapshot()
        raise IndexError(f"Step index {step_index} out of bounds (0 to {len(self._snapshots)-1}).")

    def jump_to_timestamp(self, timestamp: Any) -> Optional[ReplaySnapshot]:
        """Jump to closest snapshot matching timestamp."""
        ts_target = pd.to_datetime(timestamp)
        best_idx = 0
        min_diff = float("inf")

        for idx, snap in enumerate(self._snapshots):
            try:
                snap_ts = pd.to_datetime(snap.timestamp)
                diff = abs((snap_ts - ts_target).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_idx = idx
            except Exception:
                continue

        self._current_index = best_idx
        return self.get_current_snapshot()

    def get_timeline_summary(self) -> pd.DataFrame:
        """Return high-level DataFrame summary of all snapshots."""
        if not self._snapshots:
            return pd.DataFrame()

        data = [
            {
                "step_index": s.step_index,
                "timestamp": s.timestamp,
                "close": s.bar_data.get("close", 0.0),
                "signal": s.signal,
                "open_orders": s.open_orders_count,
                "open_positions": s.open_positions_count,
                "balance": s.balance,
                "equity": s.equity,
                "drawdown_pct": s.drawdown_pct,
            }
            for s in self._snapshots
        ]
        return pd.DataFrame(data)

    def clear(self) -> None:
        """Clear snapshots."""
        self._snapshots.clear()
        self._current_index = 0
