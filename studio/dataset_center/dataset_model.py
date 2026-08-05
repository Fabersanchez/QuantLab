"""
QuantLab Dataset Center Data Model Specification.

Defines DatasetRecord dataclass tracking dataset unique ID, name, source format (CSV, Parquet,
Feather, Arrow, HDF5, SQLite, DuckDB, PostgreSQL, ClickHouse, S3, Azure, GCS), provider, asset symbol,
market, timeframe, timezone, date range, column names, data types, row count, checksum SHA-256,
version, created_at, updated_at, author, dependencies, status, and quality score.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class DatasetRecord:
    """Institutional Dataset Center Data Model Record."""

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "MarketDataset"
    source_format: str = "CSV"  # 'CSV', 'Parquet', 'Feather', 'Arrow', 'HDF5', 'SQLite', 'DuckDB', 'PostgreSQL', 'ClickHouse', 'S3'
    provider: str = "QuantLabData"
    filepath: str = ""
    symbol: str = "EURUSD"
    market: str = "FOREX"
    timeframe: str = "1h"
    timezone: str = "UTC"
    start_date: str = ""
    end_date: str = ""
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    checksum_sha256: str = ""
    version: str = "1.0.0"
    author: str = "QuantResearcher"
    dependencies: List[str] = field(default_factory=list)
    status: str = "ACTIVE"
    quality_score: float = 100.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def update_checksum(self) -> str:
        """Compute and update SHA-256 integrity digest for dataset record."""
        raw = f"{self.dataset_id}:{self.name}:{self.version}:{self.symbol}:{self.row_count}:{self.filepath}".encode(
            "utf-8"
        )
        self.checksum_sha256 = hashlib.sha256(raw).hexdigest()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self.checksum_sha256

    def to_dict(self) -> Dict[str, Any]:
        """Convert DatasetRecord to dictionary representation."""
        self.update_checksum()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        """Reconstruct DatasetRecord from dictionary representation."""
        return cls(**data)
