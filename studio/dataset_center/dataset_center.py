"""
QuantLab Dataset Center Platform Engine.

Provides multi-format dataset registration (CSV, Parquet, Feather, Arrow, HDF5, SQLite, DuckDB,
PostgreSQL, ClickHouse, S3), deduplication enforcement, versioning, quality tracking,
and integration with Registry Platform.
"""

from typing import Any, Dict, List, Optional
from registry.dataset_registry import DatasetRegistry
from studio.dataset_center.dataset_model import DatasetRecord
from studio.events.event_bus import StudioEventBus
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("DatasetCenter")


class DatasetCenter:
    """Institutional Dataset Center Platform Engine."""

    def __init__(
        self,
        event_bus: Optional[StudioEventBus] = None,
        registry: Optional[DatasetRegistry] = None,
    ) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self.registry = registry or DatasetRegistry()
        self._datasets: Dict[str, DatasetRecord] = {}
        self._checksum_index: Dict[str, str] = {}  # sha256 -> dataset_id

    def register_dataset(
        self,
        name: str,
        source_format: str = "CSV",
        provider: str = "QuantLabData",
        filepath: str = "",
        symbol: str = "EURUSD",
        market: str = "FOREX",
        timeframe: str = "1h",
        row_count: int = 0,
        columns: Optional[List[str]] = None,
        quality_score: float = 100.0,
        author: str = "QuantResearcher",
    ) -> DatasetRecord:
        """Register dataset record with automatic deduplication check.

        Returns:
            Registered or existing DatasetRecord instance.
        """
        raw_key = f"{name}:{symbol}:{timeframe}:{row_count}:{filepath}"
        import hashlib

        chk = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        # Deduplication check
        if chk in self._checksum_index:
            existing_id = self._checksum_index[chk]
            logger.info(f"Dataset '{name}' already registered (ID={existing_id}). Preventing duplicate.")
            return self._datasets[existing_id]

        record = DatasetRecord(
            name=name,
            source_format=source_format,
            provider=provider,
            filepath=filepath,
            symbol=symbol,
            market=market,
            timeframe=timeframe,
            row_count=row_count,
            columns=columns or ["open", "high", "low", "close", "volume"],
            quality_score=quality_score,
            author=author,
        )
        record.update_checksum()

        self._datasets[record.dataset_id] = record
        self._checksum_index[chk] = record.dataset_id

        # Sync with Registry Platform
        try:
            self.registry.register_dataset(
                name=record.name,
                provider=record.provider,
                market=record.market,
                timeframe=record.timeframe,
                n_rows=record.row_count,
                checksum_sha256=record.checksum_sha256,
                version=record.version,
            )
        except Exception as e:
            logger.error(f"Registry dataset sync warning: {e}")

        logger.info(f"Registered Dataset '{name}' (ID={record.dataset_id}, Format={source_format})")
        return record

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Fetch DatasetRecord by ID."""
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[DatasetRecord]:
        """List all registered DatasetRecords."""
        return list(self._datasets.values())
