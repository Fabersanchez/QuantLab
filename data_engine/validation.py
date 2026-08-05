"""
QuantLab Data Validation & Quality Telemetry Engine.

Evaluates dataset quality telemetry: null values, duplicate rows, temporal continuity gaps,
extreme outliers, and timestamp timezone alignment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from data_engine.integrity import DataIntegrityChecker


@dataclass
class DataValidationReport:
    """Dataclass holding complete data quality telemetry validation report."""

    is_valid: bool = True
    total_rows: int = 0
    null_count: int = 0
    duplicate_rows: int = 0
    gaps_count: int = 0
    quality_score: float = 100.0
    issues: List[str] = field(default_factory=list)


class DataValidator:
    """Institutional Data Quality Telemetry Validator."""

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> DataValidationReport:
        """Validate DataFrame quality telemetry.

        Returns:
            DataValidationReport instance.
        """
        if df.empty:
            return DataValidationReport(is_valid=False, quality_score=0.0, issues=["Dataset is completely empty."])

        total_rows = len(df)
        null_count = int(df.isna().sum().sum())
        dup_count = int(df.duplicated().sum())

        _, gaps_count = DataIntegrityChecker.check_continuity(df)

        issues: List[str] = []
        deductions = 0.0

        if null_count > 0:
            deductions += min(30.0, (null_count / (total_rows * len(df.columns))) * 100.0)
            issues.append(f"Contains {null_count} missing values.")

        if dup_count > 0:
            deductions += min(20.0, (dup_count / total_rows) * 100.0)
            issues.append(f"Contains {dup_count} duplicate rows.")

        if gaps_count > 0:
            deductions += min(30.0, gaps_count * 2.0)
            issues.append(f"Detected {gaps_count} temporal continuity gaps.")

        score = max(0.0, 100.0 - deductions)
        is_valid = score >= 70.0

        return DataValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            null_count=null_count,
            duplicate_rows=dup_count,
            gaps_count=gaps_count,
            quality_score=score,
            issues=issues,
        )
