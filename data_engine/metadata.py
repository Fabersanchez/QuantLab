"""
QuantLab Dataset Metadata Specification.

Defines DatasetMetadata dataclass tracking dataset provider, market asset class, broker,
timezone, symbol, timeframe, date range, row counts, data quality score, and versioning.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, Optional


@dataclass
class DatasetMetadata:
    """Institutional Dataset Provenance Metadata."""

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "MarketDataset"
    provider: str = "QuantLabData"
    market: str = "FOREX"
    broker: str = "GenericBroker"
    timezone: str = "UTC"
    asset: str = "EURUSD"
    timeframe: str = "1h"
    start_date: str = ""
    end_date: str = ""
    row_count: int = 0
    quality_score: float = 100.0
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert DatasetMetadata to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Reconstruct DatasetMetadata from dictionary representation."""
        return cls(**data)
