"""
QuantLab Institutional Feature Store.

Centralized repository managing versioned quantitative features, schema metadata,
origins, transformations, and feature distribution statistics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


@dataclass
class FeatureMetadata:
    """Dataclass holding institutional feature specification and metadata."""

    name: str
    version: int = 1
    origin: str = "TechnicalIndicator"
    data_type: str = "float64"
    transformations: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureStore:
    """Institutional Central Feature Store Repository."""

    def __init__(self) -> None:
        """Initialize FeatureStore."""
        # Key: (feature_name, version) -> (data, FeatureMetadata)
        self._store: Dict[Tuple[str, int], Tuple[Union[pd.Series, pd.DataFrame], FeatureMetadata]] = {}
        self._latest_versions: Dict[str, int] = {}

    def register_feature(
        self,
        name: str,
        data: Union[pd.Series, pd.DataFrame],
        origin: str = "Calculated",
        transformations: Optional[List[str]] = None,
    ) -> FeatureMetadata:
        """Register a new feature or initial version into the store.

        Args:
            name: Feature identifier.
            data: pandas Series or DataFrame containing feature values.
            origin: Source origin identifier (e.g. 'TA-Lib', 'OrderFlow', 'Sentiment').
            transformations: List of transformation names applied.

        Returns:
            FeatureMetadata instance.
        """
        version = self._latest_versions.get(name, 0) + 1
        self._latest_versions[name] = version

        stats = {}
        if isinstance(data, pd.Series):
            is_num = pd.api.types.is_numeric_dtype(data)
            stats = {
                "count": len(data),
                "null_count": int(data.isnull().sum()),
                "mean": float(data.mean()) if is_num else 0.0,
                "std": float(data.std()) if is_num else 0.0,
                "min": float(data.min()) if is_num else 0.0,
                "max": float(data.max()) if is_num else 0.0,
            }
        elif isinstance(data, pd.DataFrame):
            numeric_df = data.select_dtypes(include=["number"])
            stats = {
                "count": len(data),
                "null_count": int(data.isnull().sum().sum()),
                "mean": float(numeric_df.mean().mean()) if not numeric_df.empty else 0.0,
                "std": float(numeric_df.std().mean()) if not numeric_df.empty else 0.0,
                "min": float(numeric_df.min().min()) if not numeric_df.empty else 0.0,
                "max": float(numeric_df.max().max()) if not numeric_df.empty else 0.0,
            }

        meta = FeatureMetadata(
            name=name,
            version=version,
            origin=origin,
            data_type=str(data.dtypes) if hasattr(data, "dtypes") else "numeric",
            transformations=transformations or [],
            statistics=stats,
        )

        self._store[(name, version)] = (data.copy(), meta)
        return meta

    def get_feature(
        self, name: str, version: Optional[int] = None
    ) -> Tuple[Union[pd.Series, pd.DataFrame], FeatureMetadata]:
        """Fetch feature data and metadata by name and version.

        Args:
            name: Feature name.
            version: Version integer (defaults to latest version).

        Returns:
            Tuple of (feature_data, FeatureMetadata).
        """
        if name not in self._latest_versions:
            raise KeyError(f"Feature '{name}' not found in FeatureStore.")

        v = version if version is not None else self._latest_versions[name]
        key = (name, v)
        if key not in self._store:
            raise KeyError(f"Feature '{name}' version {v} not found.")

        return self._store[key]

    def list_features(self) -> List[Dict[str, Any]]:
        """List all features currently registered in the store."""
        result = []
        for (name, v), (_, meta) in self._store.items():
            result.append(
                {
                    "name": name,
                    "version": v,
                    "origin": meta.origin,
                    "transformations": meta.transformations,
                    "statistics": meta.statistics,
                    "created_at": meta.created_at.isoformat(),
                }
            )
        return result
