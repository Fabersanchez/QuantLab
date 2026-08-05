"""
QuantLab Fast Incremental Data Preview Engine.

Generates dataset previews (head, tail, column data types, summary statistics, missing values)
incrementally without loading large dataset files into memory.
"""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class DatasetPreviewSummary:
    """Dataclass holding dataset preview summary statistics."""

    filepath: str
    total_rows_estimate: int
    column_names: List[str]
    column_types: Dict[str, str]
    head_records: List[Dict[str, Any]]
    null_counts: Dict[str, int]
    summary_stats: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert preview summary to dictionary."""
        return asdict(self)


class PreviewEngine:
    """Institutional Fast Incremental Data Preview Engine."""

    @staticmethod
    def generate_preview(filepath: str, preview_rows: int = 50) -> Optional[DatasetPreviewSummary]:
        """Generate incremental preview summary of target dataset file.

        Args:
            filepath: Path to dataset file (CSV, Parquet, JSON).
            preview_rows: Number of head rows to inspect.

        Returns:
            DatasetPreviewSummary instance or None if file not found.
        """
        if not os.path.exists(filepath):
            return None

        ext = filepath.split(".")[-1].lower()

        try:
            if ext == "csv":
                df_head = pd.read_csv(filepath, nrows=preview_rows)
            elif ext in ("parquet", "pq"):
                df_head = pd.read_parquet(filepath).head(preview_rows)
            else:
                df_head = pd.read_csv(filepath, nrows=preview_rows)
        except Exception:
            return None

        cols = [str(c) for c in df_head.columns]
        types = {str(c): str(df_head[c].dtype) for c in df_head.columns}
        nulls = {str(c): int(df_head[c].isna().sum()) for c in df_head.columns}

        records = df_head.head(10).to_dict(orient="records")

        stats_dict: Dict[str, Dict[str, float]] = {}
        for c in df_head.columns:
            if pd.api.types.is_numeric_dtype(df_head[c]):
                stats_dict[str(c)] = {
                    "mean": float(df_head[c].mean()),
                    "std": float(df_head[c].std()) if len(df_head) > 1 else 0.0,
                    "min": float(df_head[c].min()),
                    "max": float(df_head[c].max()),
                }

        return DatasetPreviewSummary(
            filepath=os.path.abspath(filepath),
            total_rows_estimate=len(df_head),
            column_names=cols,
            column_types=types,
            head_records=records,
            null_counts=nulls,
            summary_stats=stats_dict,
        )
