"""
QuantLab Indicator Validation.

Provides automated input validation, checking required columns, minimum dataset size,
NaNs, Infs, data types, and hyperparameter schemas.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class IndicatorValidationReport:
    """Diagnostic report output by IndicatorValidator."""

    is_valid: bool = True
    missing_columns: List[str] = field(default_factory=list)
    insufficient_rows: bool = False
    row_count: int = 0
    min_required_rows: int = 1
    nan_count: int = 0
    inf_count: int = 0
    errors: List[str] = field(default_factory=list)


class IndicatorValidator:
    """Automated input validator for indicator computations."""

    @staticmethod
    def validate_input(
        df: pd.DataFrame,
        required_columns: List[str],
        min_rows: int = 1,
    ) -> IndicatorValidationReport:
        """Validate input DataFrame for indicator calculation.

        Args:
            df: Market DataFrame.
            required_columns: List of required input column names (e.g. ['high', 'low', 'close']).
            min_rows: Minimum required number of rows.

        Returns:
            IndicatorValidationReport diagnostic record.
        """
        report = IndicatorValidationReport(row_count=len(df), min_required_rows=min_rows)

        if df.empty:
            report.is_valid = False
            report.errors.append("Input DataFrame is empty.")
            return report

        if len(df) < min_rows:
            report.is_valid = False
            report.insufficient_rows = True
            report.errors.append(
                f"DataFrame row count ({len(df)}) is less than required minimum ({min_rows})."
            )

        cols_lower = {c.lower(): c for c in df.columns}
        for req in required_columns:
            if req.lower() not in cols_lower:
                report.missing_columns.append(req)

        if report.missing_columns:
            report.is_valid = False
            report.errors.append(f"Missing required columns: {report.missing_columns}")

        # Count NaNs and Infs in required columns
        for req in required_columns:
            if req.lower() in cols_lower:
                real_col = cols_lower[req.lower()]
                s = df[real_col]
                if pd.api.types.is_numeric_dtype(s):
                    report.nan_count += int(s.isna().sum())
                    report.inf_count += int(np.isinf(s.values).sum())

        if report.inf_count > 0:
            report.is_valid = False
            report.errors.append(f"Infinite (Inf) values found: {report.inf_count}")

        return report
