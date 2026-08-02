"""
QuantLab Market Data Validator.

Performs institutional automated quality assurance checks on OHLCV datasets,
flagging price anomalies, timestamp irregularities, nulls, and negative volumes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ValidationReport:
    """Container for data validation diagnostics.

    Attributes:
        is_valid: Boolean indicating whether dataset passed all critical checks.
        total_rows: Number of rows in evaluated DataFrame.
        missing_columns: List of missing required columns.
        null_count: Count of null values per column.
        invalid_ohlc_count: Number of rows failing High >= Low logic.
        duplicate_timestamps: Number of duplicate timestamps detected.
        negative_volume_count: Number of rows with negative volume.
        unsorted_timestamps: Boolean flag if timestamps are not sorted.
        excessive_gaps_count: Number of identified abnormal time gaps.
        errors: Detailed log of error messages.
    """

    is_valid: bool = True
    total_rows: int = 0
    missing_columns: List[str] = field(default_factory=list)
    null_count: Dict[str, int] = field(default_factory=dict)
    invalid_ohlc_count: int = 0
    duplicate_timestamps: int = 0
    negative_volume_count: int = 0
    unsorted_timestamps: bool = False
    excessive_gaps_count: int = 0
    errors: List[str] = field(default_factory=list)


class DataValidator:
    """Institutional market data validation engine."""

    REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        max_gap_seconds: Optional[float] = None,
    ) -> ValidationReport:
        """Validate an OHLCV DataFrame against quantitative quality rules.

        Args:
            df: Market DataFrame to validate.
            timestamp_col: Name of column containing datetime timestamps.
            max_gap_seconds: Threshold for maximum acceptable time gap.

        Returns:
            ValidationReport detailing diagnostic findings.
        """
        report = ValidationReport(total_rows=len(df))

        if df.empty:
            report.is_valid = False
            report.errors.append("DataFrame is empty.")
            return report

        # 1. Missing Columns
        df_cols_lower = {c.lower(): c for c in df.columns}
        for req in self.REQUIRED_OHLCV_COLUMNS:
            if req not in df_cols_lower:
                report.missing_columns.append(req)

        if report.missing_columns:
            report.is_valid = False
            report.errors.append(f"Missing required columns: {report.missing_columns}")

        # 2. Null values
        null_series = df.isnull().sum()
        report.null_count = {
            col: int(cnt) for col, cnt in null_series.items() if cnt > 0
        }
        if report.null_count:
            report.is_valid = False
            report.errors.append(f"Null values detected: {report.null_count}")

        # 3. Timestamp checks
        if timestamp_col in df.columns:
            ts_series = pd.to_datetime(df[timestamp_col])

            # Duplicates
            dup_cnt = int(ts_series.duplicated().sum())
            report.duplicate_timestamps = dup_cnt
            if dup_cnt > 0:
                report.is_valid = False
                report.errors.append(f"Duplicate timestamps found: {dup_cnt}")

            # Sorting
            if not ts_series.is_monotonic_increasing:
                report.unsorted_timestamps = True
                report.is_valid = False
                report.errors.append(
                    "Timestamps are not strictly sorted in ascending order."
                )

            # Gaps
            if max_gap_seconds is not None and len(ts_series) > 1:
                diffs = ts_series.diff().dt.total_seconds().dropna()
                gaps = int((diffs > max_gap_seconds).sum())
                report.excessive_gaps_count = gaps
                if gaps > 0:
                    report.is_valid = False
                    report.errors.append(
                        f"Found {gaps} gaps exceeding {max_gap_seconds}s threshold."
                    )

        # 4. OHLC Logic Validation
        col_map = {c.lower(): c for c in df.columns}
        if all(k in col_map for k in ["open", "high", "low", "close"]):
            o = df[col_map["open"]]
            h = df[col_map["high"]]
            l = df[col_map["low"]]
            c = df[col_map["close"]]

            invalid_mask = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)
            invalid_cnt = int(invalid_mask.sum())
            report.invalid_ohlc_count = invalid_cnt
            if invalid_cnt > 0:
                report.is_valid = False
                report.errors.append(f"OHLC logic violated in {invalid_cnt} rows.")

        # 5. Volume check
        if "volume" in col_map:
            vol = df[col_map["volume"]]
            neg_vol = int((vol < 0).sum())
            report.negative_volume_count = neg_vol
            if neg_vol > 0:
                report.is_valid = False
                report.errors.append(f"Negative volume found in {neg_vol} rows.")

        return report
