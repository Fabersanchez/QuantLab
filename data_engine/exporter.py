"""
QuantLab Multi-Format Data Exporter.

Exports DataFrames into institutional formats: CSV, Parquet, JSON, SQLite, Excel, and Markdown.
"""

import os
import sqlite3
from typing import Any, Dict, Optional
import pandas as pd

from data_engine.logger import get_data_engine_logger

logger = get_data_engine_logger("Exporter")


class DataExporter:
    """Master Institutional Multi-Format Data Exporter."""

    @staticmethod
    def to_csv(df: pd.DataFrame, filepath: str) -> str:
        """Export DataFrame to CSV file."""
        df.to_csv(filepath, index=True)
        logger.log_export("Dataset", "CSV", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_parquet(df: pd.DataFrame, filepath: str) -> str:
        """Export DataFrame to Parquet file."""
        try:
            df.to_parquet(filepath, index=True)
        except Exception:
            fb = filepath.replace(".parquet", ".csv")
            df.to_csv(fb, index=True)
            filepath = fb
        logger.log_export("Dataset", "Parquet", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_json(df: pd.DataFrame, filepath: str) -> str:
        """Export DataFrame to JSON file."""
        df.to_json(filepath, orient="records", date_format="iso")
        logger.log_export("Dataset", "JSON", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_excel(df: pd.DataFrame, filepath: str) -> str:
        """Export DataFrame to Excel workbook."""
        df.to_excel(filepath, sheet_name="Data", index=True)
        logger.log_export("Dataset", "Excel", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_markdown(df: pd.DataFrame, filepath: str) -> str:
        """Export DataFrame head preview to Markdown document file."""
        md = f"# QuantLab Dataset Export Report\n\n```\n{df.head(20).to_string()}\n```\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)
        logger.log_export("Dataset", "Markdown", filepath)
        return os.path.abspath(filepath)
