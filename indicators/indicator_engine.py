"""
QuantLab Indicator Engine.

Main orchestrator managing indicator registration, calculation, dependencies,
caching, and execution pipeline in QuantLab.
"""

from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd

from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata
from indicators.registry import IndicatorRegistry
from indicators.validation import IndicatorValidator, IndicatorValidationReport
from indicators.indicator_pipeline import IndicatorPipeline, IndicatorCache


class IndicatorEngine:
    """Master Quantitative Indicator Engine."""

    def __init__(self) -> None:
        """Initialize IndicatorEngine subsystems."""
        self._registry = IndicatorRegistry()
        self._cache = IndicatorCache()
        self._pipeline = IndicatorPipeline(registry=self._registry, cache=self._cache)

    @property
    def registry(self) -> IndicatorRegistry:
        """Access indicator registry."""
        return self._registry

    @property
    def cache(self) -> IndicatorCache:
        """Access indicator cache."""
        return self._cache

    @property
    def pipeline(self) -> IndicatorPipeline:
        """Access indicator pipeline."""
        return self._pipeline

    def register_indicator(
        self, indicator_cls: Type[BaseIndicator], overwrite: bool = True
    ) -> None:
        """Register an indicator class into the central registry."""
        self._registry.register(indicator_cls, overwrite=overwrite)

    def calculate(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Calculate indicator by name and return output DataFrame.

        Args:
            data: Input market DataFrame.
            indicator_name: Name of registered indicator.
            params: Optional parameter overrides.

        Returns:
            pandas.DataFrame containing calculated output columns.
        """
        output_df, report = self._pipeline.run(
            data=data, indicator_name=indicator_name, params=params
        )
        if not report.is_valid:
            raise ValueError(
                f"Indicator '{indicator_name}' calculation failed validation: {report.errors}"
            )
        return output_df

    def calculate_all(
        self, data: pd.DataFrame, indicator_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate multiple indicators and attach output columns to dataset copy.

        Args:
            data: Input market DataFrame.
            indicator_names: List of indicator names or None to run all registered indicators.

        Returns:
            Combined pandas.DataFrame containing input data + indicator output columns.
        """
        names = indicator_names or self._registry.list_indicators()
        result_df = data.copy()

        for name in names:
            try:
                ind_df = self.calculate(data, name)
                if not ind_df.empty:
                    for col in ind_df.columns:
                        if col not in result_df.columns:
                            result_df[col] = ind_df[col]
            except Exception as e:
                # Log or handle uncalculable indicators for given dataset shape
                continue

        return result_df
