"""
QuantLab Active Dataset Catalog Manager.

Manages active in-memory dataset DataFrames, dataset metadata cataloging, and storage persistence.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from data_engine.metadata import DatasetMetadata
from data_engine.storage import DataStorage


class DataManager:
    """Institutional Active Dataset Catalog Manager."""

    def __init__(self, storage_dir: str = "data_storage") -> None:
        self.storage = DataStorage(storage_dir=storage_dir)
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._catalog: Dict[str, DatasetMetadata] = {}

    def register_dataset(self, name: str, df: pd.DataFrame, metadata: Optional[DatasetMetadata] = None) -> DatasetMetadata:
        """Register active dataset DataFrame and metadata."""
        meta = metadata or DatasetMetadata(name=name, row_count=len(df))
        meta.row_count = len(df)
        self._datasets[name] = df.copy()
        self._catalog[name] = meta
        return meta

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Fetch registered active dataset DataFrame."""
        df = self._datasets.get(name)
        return df.copy() if df is not None else None

    def list_datasets(self) -> List[str]:
        """List names of registered datasets."""
        return list(self._datasets.keys())
