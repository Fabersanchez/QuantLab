"""
QuantLab Dataset Registry Engine.

Registers dataset origin, provider, date ranges, market asset classes, data quality metrics,
missing values, duplicate counts, and SHA-256 checksums.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

from registry.integrity import IntegrityChecker


@dataclass
class DatasetRecord:
    """Dataclass holding dataset governance metadata."""

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "MarketDataset"
    provider: str = "QuantLabData"
    market: str = "FOREX"
    timeframe: str = "1h"
    start_date: str = ""
    end_date: str = ""
    n_rows: int = 0
    missing_values: int = 0
    duplicate_rows: int = 0
    checksum_sha256: str = ""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert DatasetRecord to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        """Reconstruct DatasetRecord from dictionary."""
        return cls(**data)


class DatasetRegistry:
    """Institutional Dataset Registry Engine."""

    def __init__(self) -> None:
        """Initialize DatasetRegistry."""
        self._datasets: Dict[str, DatasetRecord] = {}

    def register_dataset(
        self,
        name: str,
        provider: str = "QuantLabData",
        market: str = "FOREX",
        timeframe: str = "1h",
        start_date: str = "",
        end_date: str = "",
        n_rows: int = 0,
        missing_values: int = 0,
        duplicate_rows: int = 0,
        checksum_sha256: str = "",
        version: str = "1.0.0",
    ) -> DatasetRecord:
        """Register dataset record instance."""
        record = DatasetRecord(
            name=name,
            provider=provider,
            market=market,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            n_rows=n_rows,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            checksum_sha256=checksum_sha256 or IntegrityChecker.compute_sha256(f"{name}:{market}:{n_rows}"),
            version=version,
        )
        self._datasets[record.dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> Optional[DatasetRecord]:
        """Fetch DatasetRecord by ID."""
        return self._datasets.get(dataset_id)

    def list_datasets(self) -> List[DatasetRecord]:
        """List all registered datasets."""
        return list(self._datasets.values())
