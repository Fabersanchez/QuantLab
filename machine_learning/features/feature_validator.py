"""
QuantLab Feature Validator.

Performs automated feature validation, checking for NaNs, Infs, constant columns,
duplicate columns, high inter-feature correlation, invalid values, and mismatched types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class FeatureValidationReport:
    """Diagnostic report output by FeatureValidator.

    Attributes:
        is_valid: True if feature dataset passes all critical thresholds.
        total_features: Total column count evaluated.
        nan_columns: Map of column name to NaN count.
        inf_columns: Map of column name to Inf count.
        constant_columns: List of columns with zero variance.
        duplicate_columns: List of columns with duplicate data content.
        highly_correlated_pairs: List of tuple (col_a, col_b, correlation_value).
        invalid_type_columns: List of non-numeric or incompatible columns.
        warnings: Log of non-critical warnings.
        errors: Log of critical errors.
    """

    is_valid: bool = True
    total_features: int = 0
    nan_columns: Dict[str, int] = field(default_factory=dict)
    inf_columns: Dict[str, int] = field(default_factory=dict)
    constant_columns: List[str] = field(default_factory=list)
    duplicate_columns: List[str] = field(default_factory=list)
    highly_correlated_pairs: List[tuple] = field(default_factory=list)
    invalid_type_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class FeatureValidator:
    """Automated quantitative feature validator engine."""

    def __init__(self, max_correlation_threshold: float = 0.95) -> None:
        """Initialize FeatureValidator.

        Args:
            max_correlation_threshold: Threshold above which feature pairs are flagged.
        """
        self._max_corr = max_correlation_threshold

    def validate(
        self,
        df: pd.DataFrame,
        ignore_cols: Optional[List[str]] = None,
    ) -> FeatureValidationReport:
        """Validate a feature DataFrame against automated quality checks.

        Args:
            df: Input feature matrix.
            ignore_cols: Optional list of columns to exclude (e.g., 'timestamp', 'target').

        Returns:
            FeatureValidationReport object.
        """
        ignore = set(ignore_cols or [])
        target_df = df.drop(columns=[c for c in ignore if c in df.columns])

        report = FeatureValidationReport(total_features=target_df.shape[1])

        if target_df.empty:
            report.is_valid = False
            report.errors.append("Feature matrix is empty.")
            return report

        # 1. NaN, Inf, and Dtype Checks
        for col in target_df.columns:
            s = target_df[col]

            # Check Types
            if not pd.api.types.is_numeric_dtype(s):
                report.invalid_type_columns.append(col)
                report.warnings.append(
                    f"Column '{col}' has non-numeric dtype '{s.dtype}'. Categorical encoding recommended."
                )
                continue

            nan_cnt = int(s.isna().sum())
            if nan_cnt > 0:
                report.nan_columns[col] = nan_cnt

            inf_cnt = int(np.isinf(s.values).sum())
            if inf_cnt > 0:
                report.inf_columns[col] = inf_cnt

        if report.nan_columns:
            report.warnings.append(
                f"NaN values detected in {len(report.nan_columns)} features."
            )

        if report.inf_columns:
            report.is_valid = False
            report.errors.append(
                f"Infinite (Inf) values detected in {len(report.inf_columns)} features."
            )

        # 2. Constant Columns (Zero Variance)
        numeric_df = target_df.select_dtypes(include=[np.number])
        variances = numeric_df.var()
        for col, var in variances.items():
            if var == 0.0 or pd.isna(var):
                report.constant_columns.append(str(col))

        if report.constant_columns:
            report.warnings.append(
                f"Constant features detected (zero variance): {report.constant_columns}"
            )

        # 3. Duplicate Columns
        seen_hashes = {}
        for col in numeric_df.columns:
            col_bytes = numeric_df[col].values.tobytes()
            if col_bytes in seen_hashes:
                report.duplicate_columns.append(col)
            else:
                seen_hashes[col_bytes] = col

        if report.duplicate_columns:
            report.warnings.append(
                f"Duplicate feature columns detected: {report.duplicate_columns}"
            )

        # 4. High Correlation Checks
        if numeric_df.shape[1] > 1 and not numeric_df.empty:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            for col in upper_tri.columns:
                high_corr_series = upper_tri[col][
                    upper_tri[col] > self._max_corr
                ]
                for other_col, corr_val in high_corr_series.items():
                    report.highly_correlated_pairs.append(
                        (str(other_col), str(col), float(corr_val))
                    )

        if report.highly_correlated_pairs:
            report.warnings.append(
                f"Found {len(report.highly_correlated_pairs)} highly correlated feature pairs (> {self._max_corr})."
            )

        return report
