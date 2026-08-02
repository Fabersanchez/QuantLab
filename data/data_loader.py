"""
QuantLab Data Loader.

Provides modular data loading capabilities across file formats (CSV, Parquet),
relational/timeseries databases, web APIs, and broker terminals.
"""

from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd
from data.datasource import BaseDataSource


class DataLoader:
    """Modular market data loader."""

    def __init__(self, datasource: Optional[BaseDataSource] = None) -> None:
        """Initialize DataLoader with an optional default DataSource.

        Args:
            datasource: Default BaseDataSource strategy instance.
        """
        self._datasource = datasource

    def set_datasource(self, datasource: BaseDataSource) -> None:
        """Set or replace the active datasource strategy."""
        self._datasource = datasource

    def load_csv(
        self,
        file_path: Union[str, Path],
        timestamp_col: str = "timestamp",
        parse_dates: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load financial time-series data from a CSV file.

        Args:
            file_path: Absolute or relative path to CSV file.
            timestamp_col: Name of column containing datetime strings.
            parse_dates: Automatically parse timestamp_col into Datetime.
            **kwargs: Extra arguments forwarded to pandas.read_csv.

        Returns:
            pandas.DataFrame loaded from file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, **kwargs)
        if parse_dates and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        return df

    def load_parquet(
        self,
        file_path: Union[str, Path],
        columns: Optional[list] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load financial time-series data from a Parquet file.

        Args:
            file_path: Absolute or relative path to Parquet file.
            columns: Specific column list to load.
            **kwargs: Extra arguments forwarded to pandas.read_parquet.

        Returns:
            pandas.DataFrame loaded from Parquet.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return pd.read_parquet(path, columns=columns, **kwargs)

    def load_database(
        self,
        connection_or_sql: Any,
        sql_query: Optional[str] = None,
        params: Optional[Union[dict, list, tuple]] = None,
    ) -> pd.DataFrame:
        """Load data from a SQL database query using pandas.read_sql_query.

        Args:
            connection_or_sql: Connection object or SQL query string.
            sql_query: SQL query string if connection passed as first argument.
            params: Optional parameter tuple or dictionary for query formatting.

        Returns:
            pandas.DataFrame result set.
        """
        query = sql_query or connection_or_sql
        conn = connection_or_sql if sql_query else None
        return pd.read_sql_query(query, con=conn, params=params)

    def load_api(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch data from connected API data source.

        Raises:
            RuntimeError: If no DataSource is configured.
        """
        if not self._datasource:
            raise RuntimeError("No DataSource configured for API load.")
        return self._datasource.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def load_mt5(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
    ) -> pd.DataFrame:
        """Fetch data from MetaTrader 5 terminal connector.

        Raises:
            RuntimeError: If no DataSource is configured.
        """
        if not self._datasource:
            raise RuntimeError("No DataSource configured for MT5 load.")
        return self._datasource.fetch_ohlcv(
            symbol=symbol, timeframe=timeframe, limit=count
        )
