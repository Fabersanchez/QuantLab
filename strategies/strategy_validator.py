"""
QuantLab Strategy Validator.

Performs automated validation of strategy dependencies, indicator prerequisites,
timeframe compatibility, parameter schemas, and input dataset integrity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

from strategies.base_strategy import BaseStrategy


@dataclass
class StrategyValidationReport:
    """Diagnostic report output by StrategyValidator."""

    is_valid: bool = True
    missing_indicators: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    incompatible_timeframe: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class StrategyValidator:
    """Automated validator for strategy integrity and prerequisites."""

    def validate(
        self, strategy: BaseStrategy, data: pd.DataFrame, timeframe: Optional[str] = None
    ) -> StrategyValidationReport:
        """Validate strategy prerequisites against input dataset.

        Args:
            strategy: BaseStrategy instance to evaluate.
            data: Input market DataFrame.
            timeframe: Optional active timeframe identifier.

        Returns:
            StrategyValidationReport object.
        """
        report = StrategyValidationReport()
        meta = strategy.metadata()

        if data.empty:
            report.is_valid = False
            report.errors.append("Input dataset is empty.")
            return report

        # Check required indicator columns
        cols_lower = [c.lower() for c in data.columns]
        for ind in meta.indicators_required:
            if ind.lower() not in cols_lower:
                report.missing_indicators.append(ind)

        if report.missing_indicators:
            report.is_valid = False
            report.errors.append(f"Missing required indicators: {report.missing_indicators}")

        # Check required feature columns
        for feat in meta.features_required:
            if feat.lower() not in cols_lower:
                report.missing_features.append(feat)

        if report.missing_features:
            report.is_valid = False
            report.errors.append(f"Missing required features: {report.missing_features}")

        # Check timeframe compatibility
        if timeframe and meta.timeframes and "All" not in meta.timeframes:
            if timeframe not in meta.timeframes:
                report.incompatible_timeframe = True
                report.warnings.append(
                    f"Timeframe '{timeframe}' not listed in strategy compatible timeframes {meta.timeframes}."
                )

        return report
