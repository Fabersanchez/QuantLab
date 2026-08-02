"""
QuantLab Walk Forward Window Generator Abstractions and Data Structures.

Defines WindowSplit dataclass, BaseWindowGenerator abstract class, and WindowGeneratorFactory.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class WindowSplit:
    """Dataclass representing an individual Walk Forward train/validation split window."""

    window_index: int
    train_start_index: int
    train_end_index: int
    val_start_index: int
    val_end_index: int
    train_start_timestamp: Any = None
    train_end_timestamp: Any = None
    val_start_timestamp: Any = None
    val_end_timestamp: Any = None

    @property
    def train_bars(self) -> int:
        """Return number of bars in training window."""
        return self.train_end_index - self.train_start_index + 1

    @property
    def val_bars(self) -> int:
        """Return number of bars in validation window."""
        return self.val_end_index - self.val_start_index + 1


class BaseWindowGenerator(ABC):
    """Abstract Base Class for all Walk Forward window generator algorithms."""

    @abstractmethod
    def generate_windows(self, data: pd.DataFrame) -> List[WindowSplit]:
        """Generate list of WindowSplit objects for target dataset.

        Args:
            data: Input market DataFrame.

        Returns:
            List of WindowSplit objects.
        """
        pass


class WindowGeneratorFactory:
    """Factory to instantiate window generators by type identifier."""

    @staticmethod
    def create(window_type: str, **kwargs) -> BaseWindowGenerator:
        """Create window generator instance.

        Args:
            window_type: Type identifier ('rolling', 'sliding', 'expanding', 'anchored', 'custom').
            kwargs: Constructor keyword arguments.

        Returns:
            Instance of BaseWindowGenerator.
        """
        w_type = window_type.lower().strip()

        if w_type in ("rolling", "sliding"):
            from walk_forward.rolling_windows import RollingWindowGenerator
            return RollingWindowGenerator(**kwargs)
        elif w_type == "expanding":
            from walk_forward.expanding_windows import ExpandingWindowGenerator
            return ExpandingWindowGenerator(**kwargs)
        elif w_type in ("anchored", "custom"):
            from walk_forward.anchored_windows import AnchoredWindowGenerator, CustomWindowGenerator
            if w_type == "anchored":
                return AnchoredWindowGenerator(**kwargs)
            else:
                return CustomWindowGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown window generator type '{window_type}'.")
