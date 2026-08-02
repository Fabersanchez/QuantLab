"""
QuantLab Expanding Walk Forward Window Generator.

Generates expanding training windows where the training start date remains fixed while the training end date
expands forward, followed by a sliding validation window.
"""

from typing import List, Optional
import pandas as pd

from walk_forward.window_generator import BaseWindowGenerator, WindowSplit


class ExpandingWindowGenerator(BaseWindowGenerator):
    """Expanding Window Generator (fixed start index, growing train window, sliding validation window)."""

    def __init__(
        self, initial_train_bars: int = 252, val_bars: int = 63, step_bars: Optional[int] = None
    ) -> None:
        """Initialize ExpandingWindowGenerator.

        Args:
            initial_train_bars: Starting size of initial training window in bars.
            val_bars: Size of validation window in bars.
            step_bars: Step expansion size per iteration (defaults to val_bars).
        """
        if initial_train_bars <= 0 or val_bars <= 0:
            raise ValueError("initial_train_bars and val_bars must be positive integers.")

        self.initial_train_bars = int(initial_train_bars)
        self.val_bars = int(val_bars)
        self.step_bars = int(step_bars) if step_bars is not None and step_bars > 0 else int(val_bars)

    def generate_windows(self, data: pd.DataFrame) -> List[WindowSplit]:
        """Generate expanding WindowSplit list."""
        total_rows = len(data)
        min_required = self.initial_train_bars + self.val_bars
        if total_rows < min_required:
            raise ValueError(
                f"Dataset length ({total_rows}) is shorter than minimum required ({min_required})."
            )

        windows: List[WindowSplit] = []
        current_train_end = self.initial_train_bars - 1
        window_counter = 0

        while current_train_end + self.val_bars < total_rows:
            train_start = 0  # Fixed start
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
