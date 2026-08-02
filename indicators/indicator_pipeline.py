"""
QuantLab Indicator Pipeline and Cache.

Implements sequential indicator execution (Validation -> Dependency Resolution -> Calculation -> Cache -> Output)
and in-memory calculation caching to prevent redundant computations.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from indicators.base_indicator import BaseIndicator
from indicators.registry import IndicatorRegistry
from indicators.validation import IndicatorValidator, IndicatorValidationReport


class IndicatorCache:
    """In-memory cache for storing pre-calculated indicator DataFrames."""

    def __init__(self, max_entries: int = 200) -> None:
        self._max_entries = max_entries
        self._cache: Dict[str, pd.DataFrame] = {}

    def _generate_key(
        self, data: pd.DataFrame, indicator_name: str, params: Dict[str, Any]
    ) -> str:
        # Create deterministic hash key from data shape, head/tail close price, indicator name, and params
        data_sig = f"{len(data)}_{data.index[0] if len(data) > 0 else 0}_{data.index[-1] if len(data) > 0 else 0}"
        if "close" in [c.lower() for c in data.columns]:
            c_col = [c for c in data.columns if c.lower() == "close"][0]
            data_sig += f"_{data[c_col].iloc[0] if len(data) > 0 else 0}_{data[c_col].iloc[-1] if len(data) > 0 else 0}"

        params_sig = str(sorted(params.items()))
        raw_key = f"{indicator_name.lower()}:{data_sig}:{params_sig}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(
        self, data: pd.DataFrame, indicator_name: str, params: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        key = self._generate_key(data, indicator_name, params)
        return self._cache.get(key)

    def set(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Dict[str, Any],
        result: pd.DataFrame,
    ) -> None:
        if len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))
        key = self._generate_key(data, indicator_name, params)
        self._cache[key] = result.copy()

    def clear(self) -> None:
        self._cache.clear()


class IndicatorPipeline:
    """Configurable execution pipeline for quantitative indicators."""

    def __init__(
        self,
        registry: IndicatorRegistry,
        cache: Optional[IndicatorCache] = None,
    ) -> None:
        self._registry = registry
        self._cache = cache or IndicatorCache()
        self._validator = IndicatorValidator()

    @property
    def cache(self) -> IndicatorCache:
        return self._cache

    def run(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, IndicatorValidationReport]:
        """Run indicator calculation through pipeline sequence.

        Pipeline Steps:
        1. Lookup: Get indicator class from registry.
        2. Instantiate: Instantiate indicator with parameters.
        3. Validation: Verify input schema and min rows.
        4. Dependency Resolution: Ensure required dependencies exist in input.
        5. Cache Check: Return cached DataFrame if present.
        6. Calculation: Compute indicator.
        7. Cache Store: Store result in cache.

        Args:
            data: Input market DataFrame.
            indicator_name: Identifier of indicator.
            params: Optional parameter overrides.

        Returns:
            Tuple of (Output DataFrame containing computed indicators, ValidationReport).
        """
        ind_cls = self._registry.get(indicator_name)
        indicator = ind_cls(params=params)
        meta = indicator.metadata()

        # Step 1: Validation
        report = self._validator.validate_input(
            df=data,
            required_columns=meta.dependencies,
            min_rows=meta.parameters.get("period", 1),
        )

        if not report.is_valid:
            return pd.DataFrame(index=data.index), report

        # Step 2: Cache Check
        cached_df = self._cache.get(data, meta.name, indicator.params)
        if cached_df is not None:
            return cached_df, report

        # Step 3: Calculation
        result_df = indicator.calculate(data)

        # Step 4: Cache Store
        self._cache.set(data, meta.name, indicator.params, result_df)

        return result_df, report
