"""
QuantLab Data Source Adapters.

Provides abstract BaseDataSource and concrete adapters for CSV, Parquet, SQLite, DuckDB,
REST APIs, and Brokers.
"""

from abc import ABC, abstractmethod
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseDataSource(ABC):
    """Abstract Base Class for Data Source Adapters."""

    def __init__(self, name: str, source_type: str) -> None:
        self.name = name
        self.source_type = source_type

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch DataFrame for target symbol and date range."""
        pass


class CSVDataSource(BaseDataSource):
    """CSV File Data Source Adapter."""

    def __init__(self, filepath: str) -> None:
        super().__init__("CSVDataSource", "CSV")
        self.filepath = filepath

    def fetch_data(self, symbol: str = "", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        if not self.filepath:
            return pd.DataFrame()
        df = pd.read_csv(self.filepath)
        return df


class ParquetDataSource(BaseDataSource):
    """Parquet File Data Source Adapter."""

    def __init__(self, filepath: str) -> None:
        super().__init__("ParquetDataSource", "Parquet")
        self.filepath = filepath

    def fetch_data(self, symbol: str = "", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        if not self.filepath:
            return pd.DataFrame()
        return pd.read_parquet(self.filepath)


class SQLiteDataSource(BaseDataSource):
    """SQLite Database Data Source Adapter."""

    def __init__(self, db_path: str, table_name: str = "market_data") -> None:
        super().__init__("SQLiteDataSource", "SQLite")
        self.db_path = db_path
        self.table_name = table_name

    def fetch_data(self, symbol: str = "", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            query = f"SELECT * FROM {self.table_name}"
            if symbol:
                query += f" WHERE symbol = '{symbol}'"
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
