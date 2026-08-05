"""
QuantLab Data Schema Enforcement Engine.

Validates DataFrame column names, expected OHLCV schema rules, required data types,
and non-empty dataset requirements.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class SchemaValidator:
    """Institutional Data Schema Enforcement Engine."""

    REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close"]

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """Validate DataFrame contains all required columns.

        Args:
            df: DataFrame to validate.
            required_columns: Optional list of required column names.

        Returns:
            Tuple of (is_valid: bool, missing_columns: List[str]).
        """
        reqs = required_columns or SchemaValidator.REQUIRED_OHLCV_COLUMNS
        if df.empty:
            return False, reqs

        cols_lower = [str(c).lower() for c in df.columns]
        missing = [c for c in reqs if c.lower() not in cols_lower]
        return len(missing) == 0, missing

    @staticmethod
    def enforce_ohlcv_types(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce standard numeric and timestamp types on OHLCV columns."""
        df_out = df.copy()
        cols_lower_map = {str(c).lower(): c for c in df_out.columns}

        for col in ["open", "high", "low", "close", "volume"]:
            if col in cols_lower_map:
                actual_col = cols_lower_map[col]
                df_out[actual_col] = pd.to_numeric(df_out[actual_col], errors="coerce")

        if "timestamp" in cols_lower_map:
            ts_col = cols_lower_map["timestamp"]
            df_out[ts_col] = pd.to_datetime(df_out[ts_col], errors="coerce")

        return df_out
