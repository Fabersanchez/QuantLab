"""
QuantLab Market Dataset Container.

Encapsulates market data, metadata (asset, timeframe, broker, timezone),
feature/target designations, and dataset statistical summaries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class MarketMetadata:
    """Metadata container for MarketDataset."""

    asset: str
    timeframe: str
    broker: str = "Generic"
    time_zone: str = "UTC"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


class MarketDataset:
    """Institutional Market Dataset Container."""

    def __init__(
        self,
        data: pd.DataFrame,
        asset: str,
        timeframe: str,
        broker: str = "Generic",
        time_zone: str = "UTC",
        features: Optional[List[str]] = None,
        target: Optional[str] = None,
    ) -> None:
        """Initialize MarketDataset.

        Args:
            data: Raw pandas DataFrame containing market features and prices.
            asset: Symbol name (e.g., 'EURUSD', 'AAPL').
            timeframe: Candle bar timeframe (e.g., '1m', '1h').
            broker: Source broker or exchange name.
            time_zone: Timezone identifier.
            features: Designated feature column names.
            target: Designated target column name.
        """
        self._data: pd.DataFrame = data.copy()
        self._metadata = MarketMetadata(
            asset=asset, timeframe=timeframe, broker=broker, time_zone=time_zone
        )
        self._features: List[str] = features or []
        self._target: Optional[str] = target

    @property
    def data(self) -> pd.DataFrame:
        """Return underlying pandas DataFrame."""
        return self._data

    @property
    def metadata(self) -> MarketMetadata:
        """Return dataset metadata container."""
        return self._metadata

    @property
    def rows(self) -> int:
        """Return number of rows in dataset."""
        return len(self._data)

    @property
    def features(self) -> List[str]:
        """Return list of designated feature column names."""
        return self._features

    @features.setter
    def features(self, cols: List[str]) -> None:
        self._features = cols

    @property
    def target(self) -> Optional[str]:
        """Return target column name."""
        return self._target

    @target.setter
    def target(self, col: str) -> None:
        self._target = col

    def summary_statistics(self) -> Dict[str, Any]:
        """Compute statistical summary of numerical features."""
        numeric_df = self._data.select_dtypes(include=["number"])
        return {
            "row_count": self.rows,
            "columns": list(self._data.columns),
            "means": numeric_df.mean().to_dict(),
            "stds": numeric_df.std().to_dict(),
            "mins": numeric_df.min().to_dict(),
            "maxs": numeric_df.max().to_dict(),
        }
