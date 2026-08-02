"""Strategy Signal Confirmations."""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseConfirmation(ABC):
    """Abstract Base Class for signal confirmation rules."""

    @abstractmethod
    def confirm(self, data: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        """Confirm or reject raw trading signal."""
        pass


class MultiIndicatorConfirmation(BaseConfirmation):
    """Confirms signal if secondary indicator agrees."""

    def __init__(self, confirmation_col: str = "rsi_14", min_val: float = 50.0, signal_direction: int = 1) -> None:
        self.col = confirmation_col
        self.min_val = min_val
        self.signal_dir = signal_direction

    def confirm(self, data: pd.DataFrame, raw_signal: pd.Series) -> pd.Series:
        if self.col not in data.columns:
            return raw_signal

        ind = data[self.col]
        confirmed = raw_signal.copy()

        if self.signal_dir == 1:
            invalid_mask = (raw_signal == 1) & (ind < self.min_val)
            confirmed[invalid_mask] = 0
        elif self.signal_dir == -1:
            invalid_mask = (raw_signal == -1) & (ind > self.min_val)
            confirmed[invalid_mask] = 0

        return confirmed
