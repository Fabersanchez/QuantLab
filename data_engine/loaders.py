"""
QuantLab Data Loader System.

Loads CSV, Parquet, JSON, and SQLite files into pandas DataFrames with automated type casting.
"""

import os
from typing import Any, Dict, Optional
import pandas as pd

from data_engine.schema import SchemaValidator


class DataLoader:
    """Institutional Data Loader System."""

    @staticmethod
    def load_file(filepath: str) -> pd.DataFrame:
        """Load data file based on extension and enforce schema types."""
        if not os.path.exists(filepath):
            return pd.DataFrame()

        ext = filepath.split(".")[-1].lower()
        if ext == "csv":
            df = pd.read_csv(filepath)
        elif ext in ("parquet", "pq"):
            df = pd.read_parquet(filepath)
        elif ext == "json":
            df = pd.read_json(filepath)
        else:
            df = pd.read_csv(filepath)

        return SchemaValidator.enforce_ohlcv_types(df)
