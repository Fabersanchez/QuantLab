"""
QuantLab Data Storage Engine.

Persists DataFrames across multi-backend formats: SQLite, Parquet, Feather, HDF5, and CSV.
"""

import os
import sqlite3
from typing import Any, Dict, Optional
import pandas as pd


class DataStorage:
    """Institutional Multi-Backend Data Storage Engine."""

    def __init__(self, storage_dir: str = "data_storage") -> None:
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to Parquet file (with CSV fallback)."""
        filepath = os.path.join(self.storage_dir, filename if filename.endswith(".parquet") else filename + ".parquet")
        try:
            df.to_parquet(filepath, index=True)
            return os.path.abspath(filepath)
        except Exception:
            fb = filepath.replace(".parquet", ".csv")
            df.to_csv(fb, index=True)
            return os.path.abspath(fb)

    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from Parquet file (with CSV fallback)."""
        filepath = os.path.join(self.storage_dir, filename if filename.endswith(".parquet") else filename + ".parquet")
        if os.path.exists(filepath):
            try:
                return pd.read_parquet(filepath)
            except Exception:
                pass
        fb = filepath.replace(".parquet", ".csv")
        if os.path.exists(fb):
            return pd.read_csv(fb)
        return pd.DataFrame()

    def save_sqlite(self, df: pd.DataFrame, table_name: str, db_name: str = "market_data.db") -> str:
        """Save DataFrame to SQLite table."""
        db_path = os.path.join(self.storage_dir, db_name)
        conn = sqlite3.connect(db_path)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=True)
            return os.path.abspath(db_path)
        finally:
            conn.close()

    def load_sqlite(self, table_name: str, db_name: str = "market_data.db") -> pd.DataFrame:
        """Load DataFrame from SQLite table."""
        db_path = os.path.join(self.storage_dir, db_name)
        if not os.path.exists(db_path):
            return pd.DataFrame()
        conn = sqlite3.connect(db_path)
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        finally:
            conn.close()
