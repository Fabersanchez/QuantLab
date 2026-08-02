"""
QuantLab Base Indicator Interface.

Defines the abstract BaseIndicator interface from which all technical,
market structure, statistical, and custom quantitative indicators inherit.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from indicators.metadata import IndicatorMetadata


class BaseIndicator(ABC):
    """Abstract Base Class for all quantitative indicators in QuantLab."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize indicator instance with optional hyperparameter overrides.

        Args:
            params: Dictionary of parameters overriding default configuration.
        """
        self._params: Dict[str, Any] = self.default_parameters()
        if params:
            self._params.update(params)

    @classmethod
    @abstractmethod
    def metadata(cls) -> IndicatorMetadata:
        """Return the IndicatorMetadata specification for this class."""
        pass

    @classmethod
    def default_parameters(cls) -> Dict[str, Any]:
        """Return default parameter dictionary defined in metadata."""
        return cls.metadata().parameters.copy()

    @property
    def params(self) -> Dict[str, Any]:
        """Return active parameters dictionary."""
        return self._params

    def initialize(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Re-initialize indicator instance parameters."""
        if params:
            self._params.update(params)

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute indicator metrics and return DataFrame containing output columns.

        Args:
            data: Input OHLCV or market feature DataFrame.

        Returns:
            pandas.DataFrame containing computed indicator output columns.
        """
        pass

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Verify input DataFrame meets column prerequisites and row length."""
        if data.empty:
            return False
        meta = self.metadata()
        for dep in meta.dependencies:
            if dep.lower() not in [c.lower() for c in data.columns]:
                return False
        return True

    def reset(self) -> None:
        """Reset internal state or buffers if stateful."""
        pass
