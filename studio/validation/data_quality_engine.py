"""
QuantLab Data Quality Platform Engine.

Analyses null values, duplicate rows, outliers, temporal gaps, invalid types, constant columns,
extreme correlations, quality scores (0-100), and generates automatic quality alert telemetry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from data_engine.validation import DataValidationReport, DataValidator
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("DataQualityEngine")


@dataclass
class QualityAlert:
    """Dataclass holding automated quality alert telemetry item."""

    alert_id: str
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    issue_type: str
    description: str


class DataQualityEngine:
    """Institutional Data Quality Platform Engine."""

    @staticmethod
    def evaluate_quality(df: pd.DataFrame) -> Tuple[DataValidationReport, List[QualityAlert]]:
        """Run deep data quality evaluation and return report & quality alerts.

        Returns:
            Tuple of (DataValidationReport, List[QualityAlert]).
        """
        report = DataValidator.validate_dataset(df)
        alerts: List[QualityAlert] = []

        if df.empty:
            alerts.append(QualityAlert("ALT-001", "CRITICAL", "EmptyDataset", "Dataset contains 0 rows."))
            return report, alerts

        # Check constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                alerts.append(
                    QualityAlert(
                        f"ALT-CONST-{col}",
                        "WARNING",
                        "ConstantColumn",
                        f"Column '{col}' is constant (zero variance).",
                    )
                )

        # Check null count threshold
        if report.null_count > 0:
            alerts.append(
                QualityAlert(
                    "ALT-NULLS",
                    "WARNING" if report.null_count < 100 else "CRITICAL",
                    "MissingValues",
                    f"Found {report.null_count} missing values.",
                )
            )

        # Check temporal continuity gaps
        if report.gaps_count > 0:
            alerts.append(
                QualityAlert(
                    "ALT-GAPS",
                    "CRITICAL",
                    "TemporalContinuityGaps",
                    f"Detected {report.gaps_count} temporal gaps in market data continuity.",
                )
            )

        logger.info(f"Evaluated Data Quality: Score={report.quality_score:.1f}/100 | Alerts={len(alerts)}")
        return report, alerts
