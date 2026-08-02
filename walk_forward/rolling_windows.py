"""
QuantLab Rolling and Sliding Walk Forward Window Generators.

Generates fixed-size training windows and validation windows that roll forward in fixed steps across time.
"""

from typing import List, Optional
import pandas as pd

from walk_forward.window_generator import BaseWindowGenerator, WindowSplit


class RollingWindowGenerator(BaseWindowGenerator):
    """Rolling Window Generator (fixed train size, fixed validation size, fixed step shift)."""

    def __init__(
        self, train_bars: int = 252, val_bars: int = 63, step_bars: Optional[int] = None
    ) -> None:
        """Initialize RollingWindowGenerator.

        Args:
            train_bars: Number of bar periods in training (In-Sample) window.
            val_bars: Number of bar periods in validation (Out-of-Sample) window.
            step_bars: Step size in bars to roll forward per iteration (defaults to val_bars).
        """
        if train_bars <= 0 or val_bars <= 0:
            raise ValueError("train_bars and val_bars must be positive integers.")

        self.train_bars = int(train_bars)
        self.val_bars = int(val_bars)
        self.step_bars = int(step_bars) if step_bars is not None and step_bars > 0 else int(val_bars)

    def generate_windows(self, data: pd.DataFrame) -> List[WindowSplit]:
        """Generate rolling WindowSplit list."""
        total_rows = len(data)
        min_required = self.train_bars + self.val_bars
        if total_rows < min_required:
            raise ValueError(
                f"Dataset length ({total_rows}) is shorter than minimum required window size ({min_required})."
            )

        windows: List[WindowSplit] = []
        idx = 0
        window_counter = 0

        while idx + self.train_bars + self.val_bars <= total_rows:
            train_start = idx
            train_end = idx + self.train_bars - 1
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
            idx += self.step_bars

        return windows


class SlidingWindowGenerator(RollingWindowGenerator):
    """Sliding Window Generator (alias with custom step overlap)."""

    def __init__(self, train_bars: int = 252, val_bars: int = 63, overlap_bars: int = 0) -> None:
        """Initialize SlidingWindowGenerator.

        Args:
            train_bars: Training window size.
            val_bars: Validation window size.
            overlap_bars: Overlap between consecutive validation windows.
        """
        step = max(1, val_bars - overlap_bars)
        super().__init__(train_bars=train_bars, val_bars=val_bars, step_bars=step)
