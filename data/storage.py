"""
QuantLab Storage Abstraction.

Provides standardized persistence adapters for reading and writing financial datasets
across multiple formats (CSV, Parquet, Feather, HDF5, SQLite, PostgreSQL).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import sqlite3
from typing import Any, Optional, Union
import pandas as pd


class BaseStorage(ABC):
    """Abstract interface for dataset persistence adapters."""

    @abstractmethod
    def save(
        self, df: pd.DataFrame, target: Union[str, Path], **kwargs: Any
    ) -> bool:
        """Persist DataFrame to storage target."""
        pass

    @abstractmethod
    def load(self, target: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        """Load DataFrame from storage target."""
        pass


class CSVStorage(BaseStorage):
    """CSV Persistence Adapter."""

    def save(
        self,
        df: pd.DataFrame,
        target: Union[str, Path],
        index: bool = False,
        **kwargs: Any,
    ) -> bool:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index, **kwargs)
        return True

    def load(self, target: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(target, **kwargs)


class ParquetStorage(BaseStorage):
    """Parquet Persistence Adapter."""

    def save(
        self,
        df: pd.DataFrame,
        target: Union[str, Path],
        index: bool = False,
        **kwargs: Any,
    ) -> bool:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=index, **kwargs)
        return True

    def load(self, target: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        return pd.read_parquet(target, **kwargs)


class FeatherStorage(BaseStorage):
    """Feather Persistence Adapter."""

    def save(
        self, df: pd.DataFrame, target: Union[str, Path], **kwargs: Any
    ) -> bool:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(path, **kwargs)
        return True

    def load(self, target: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
        return pd.read_feather(target, **kwargs)


class HDF5Storage(BaseStorage):
    """HDF5 Persistence Adapter."""

    def save(
        self,
        df: pd.DataFrame,
        target: Union[str, Path],
        key: str = "market_data",
        **kwargs: Any,
    ) -> bool:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_hdf(path, key=key, mode="w", **kwargs)
        return True

    def load(
        self, target: Union[str, Path], key: str = "market_data", **kwargs: Any
    ) -> pd.DataFrame:
        return pd.read_hdf(target, key=key, **kwargs)


class SQLiteStorage(BaseStorage):
    """SQLite Database Persistence Adapter."""

    def save(
        self,
        df: pd.DataFrame,
        target: Union[str, Path],
        table_name: str = "market_data",
        if_exists: str = "replace",
        index: bool = False,
        **kwargs: Any,
    ) -> bool:
        conn = sqlite3.connect(str(target))
        df.to_sql(table_name, con=conn, if_exists=if_exists, index=index, **kwargs)
        conn.close()
        return True

    def load(
        self,
        target: Union[str, Path],
        table_name: str = "market_data",
        sql_query: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        conn = sqlite3.connect(str(target))
        query = sql_query or f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, con=conn, **kwargs)
        conn.close()
        return df


class StorageFactory:
    """Factory for acquiring storage persistence instances."""

    _STORAGE_MAP = {
        "csv": CSVStorage,
        "parquet": ParquetStorage,
        "feather": FeatherStorage,
        "hdf5": HDF5Storage,
        "sqlite": SQLiteStorage,
    }

    @classmethod
    def get_storage(cls, format_name: str) -> BaseStorage:
        fmt = format_name.lower()
        if fmt not in cls._STORAGE_MAP:
            raise ValueError(f"Unsupported storage format: {format_name}")
        return cls._STORAGE_MAP[fmt]()
