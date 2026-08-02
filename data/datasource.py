"""
QuantLab Data Source Interface.

Defines the abstract interface for all market data sources (files, databases,
REST APIs, WebSockets, broker terminals, third-party quantitative feeds).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseDataSource(ABC):
    """Abstract Base Class for all quantitative data sources."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the unique identifier string for the data source."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and clean up resources."""
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV market data.

        Args:
            symbol: Financial instrument symbol (e.g., 'EURUSD', 'BTC/USD').
            timeframe: Bar aggregation period (e.g., '1m', '5m', '1h', '1d').
            start_time: Start timestamp bound.
            end_time: End timestamp bound.
            limit: Maximum number of rows to retrieve.

        Returns:
            pandas.DataFrame with standardized OHLCV schema.
        """
        pass


class MockDataSource(BaseDataSource):
    """Mock Data Source for testing and decoupling demonstration."""

    def __init__(self, name: str = "MockFeed") -> None:
        self._name = name
        self._connected = False

    @property
    def source_name(self) -> str:
        return self._name

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        n_rows = limit or 100
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_rows, freq="1min")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1000.0,
            }
        )
        return df
