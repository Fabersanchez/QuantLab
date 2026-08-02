"""
QuantLab Anchored and Custom Walk Forward Window Generators.

Provides AnchoredWindowGenerator (training window anchored to a fixed index/date with expanding or rolling windows)
and CustomWindowGenerator (explicit user-specified window bounds).
"""

from typing import Any, List, Optional, Tuple
import pandas as pd

from walk_forward.window_generator import BaseWindowGenerator, WindowSplit


class AnchoredWindowGenerator(BaseWindowGenerator):
    """Anchored Window Generator (anchored starting point with growing training windows)."""

    def __init__(
        self,
        anchor_index: int = 0,
        initial_train_bars: int = 252,
        val_bars: int = 63,
        step_bars: Optional[int] = None,
    ) -> None:
        """Initialize AnchoredWindowGenerator.

        Args:
            anchor_index: Starting anchor bar index.
            initial_train_bars: Minimum initial training window size.
            val_bars: Validation window size.
            step_bars: Step size to advance.
        """
        self.anchor_index = max(0, int(anchor_index))
        self.initial_train_bars = int(initial_train_bars)
        self.val_bars = int(val_bars)
        self.step_bars = int(step_bars) if step_bars is not None and step_bars > 0 else int(val_bars)

    def generate_windows(self, data: pd.DataFrame) -> List[WindowSplit]:
        """Generate anchored WindowSplit list."""
        total_rows = len(data)
        min_required = self.anchor_index + self.initial_train_bars + self.val_bars
        if total_rows < min_required:
            raise ValueError(
                f"Dataset length ({total_rows}) is shorter than required anchored bounds ({min_required})."
            )

        windows: List[WindowSplit] = []
        current_train_end = self.anchor_index + self.initial_train_bars - 1
        window_counter = 0

        while current_train_end + self.val_bars < total_rows:
            train_start = self.anchor_index
            train_end = current_train_end
            val_start = train_end + 1
            val_end = min(val_start + self.val_bars - 1, total_rows - 1)

            t_start_ts = data.index[train_start] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[train_start].get("timestamp", train_start)
            t_end_ts = data.index[train_end] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[train_end].get("timestamp", train_end)
            v_start_ts = data.index[val_start] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[val_start].get("timestamp", val_start)
            v_end_ts = data.index[val_end] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[val_end].get("timestamp", val_end)

            split = WindowSplit(
                window_index=window_counter,
                train_start_index=train_start,
                train_end_index=train_end,
                val_start_index=val_start,
                val_end_index=val_end,
                train_start_timestamp=t_start_ts,
                train_end_timestamp=t_end_ts,
                val_start_timestamp=v_start_ts,
                val_end_timestamp=v_end_ts,
            )

            windows.append(split)
            window_counter += 1
            current_train_end += self.step_bars

        return windows


class CustomWindowGenerator(BaseWindowGenerator):
    """Custom Window Generator accepting explicit tuple bounds."""

    def __init__(self, explicit_bounds: List[Tuple[int, int, int, int]]) -> None:
        """Initialize CustomWindowGenerator.

        Args:
            explicit_bounds: List of (train_start_idx, train_end_idx, val_start_idx, val_end_idx) tuples.
        """
        if not explicit_bounds:
            raise ValueError("explicit_bounds must be a non-empty list of 4-tuples.")
        self.explicit_bounds = explicit_bounds

    def generate_windows(self, data: pd.DataFrame) -> List[WindowSplit]:
        """Generate custom WindowSplit list."""
        total_rows = len(data)
        windows: List[WindowSplit] = []

        for idx, (t_start, t_end, v_start, v_end) in enumerate(self.explicit_bounds):
            if v_end >= total_rows:
                raise IndexError(f"Validation end index {v_end} exceeds dataset length {total_rows}.")

            t_start_ts = data.index[t_start] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[t_start].get("timestamp", t_start)
            t_end_ts = data.index[t_end] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[t_end].get("timestamp", t_end)
            v_start_ts = data.index[v_start] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[v_start].get("timestamp", v_start)
            v_end_ts = data.index[v_end] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[v_end].get("timestamp", v_end)

            split = WindowSplit(
                window_index=idx,
                train_start_index=t_start,
                train_end_index=t_end,
                val_start_index=v_start,
                val_end_index=v_end,
                train_start_timestamp=t_start_ts,
                train_end_timestamp=t_end_ts,
                val_start_timestamp=v_start_ts,
                val_end_timestamp=v_end_ts,
            )

            windows.append(split)

        return windows
