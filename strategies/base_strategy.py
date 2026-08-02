"""
QuantLab Base Strategy Interface.

Defines the abstract BaseStrategy interface from which all quantitative,
algorithmic, and machine learning trading strategies inherit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from strategies.strategy_metadata import StrategyMetadata


class BaseStrategy(ABC):
    """Abstract Base Class for all quantitative strategies in QuantLab."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize strategy instance with hyperparameter overrides.

        Args:
            params: Dictionary of hyperparameter overrides.
        """
        self._params: Dict[str, Any] = self.default_parameters()
        if params:
            self._params.update(params)

    @classmethod
    @abstractmethod
    def metadata(cls) -> StrategyMetadata:
        """Return StrategyMetadata specification for this class."""
        pass

    @classmethod
    def default_parameters(cls) -> Dict[str, Any]:
        """Return default parameter dictionary defined in metadata."""
        return {}

    @property
    def params(self) -> Dict[str, Any]:
        """Return active strategy parameters."""
        return self._params

    def initialize(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Re-initialize strategy parameters."""
        if params:
            self._params.update(params)

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataset, calculating necessary indicators or features.

        Args:
            data: Raw input market DataFrame.

        Returns:
            Prepared DataFrame with attached indicators and features.
        """
        return data.copy()

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signal series (+1 Long, -1 Short, 0 Neutral).

        Args:
            data: Prepared market DataFrame.

        Returns:
            pandas.DataFrame containing a 'signal' column.
        """
        pass

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate input DataFrame schema and required indicators."""
        if data.empty:
            return False
        meta = self.metadata()
        data_cols_lower = [c.lower() for c in data.columns]
        for ind in meta.indicators_required:
            if ind.lower() not in data_cols_lower:
                return False
        return True

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute full strategy logic sequence: prepare -> validate -> generate_signal."""
        prepared_df = self.prepare(data)
        if not self.validate(prepared_df):
            raise ValueError(f"Data validation failed for strategy '{self.metadata().name}'.")
        signal_df = self.generate_signal(prepared_df)
        return pd.concat([prepared_df, signal_df], axis=1)

    def reset(self) -> None:
        """Reset internal state or stateful variables."""
        pass
