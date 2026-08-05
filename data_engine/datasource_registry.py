"""
QuantLab Data Source Registry Engine.

Registers and manages available data sources and provider connectors.
"""

from typing import Dict, List, Optional
from data_engine.datasource import BaseDataSource


class DataSourceRegistry:
    """Institutional Data Source Registry Engine."""

    def __init__(self) -> None:
        self._sources: Dict[str, BaseDataSource] = {}

    def register_source(self, source_id: str, source: BaseDataSource) -> None:
        """Register a data source instance."""
        self._sources[source_id.lower()] = source

    def get_source(self, source_id: str) -> Optional[BaseDataSource]:
        """Fetch registered data source by ID."""
        return self._sources.get(source_id.lower())

    def list_sources(self) -> List[str]:
        """List registered source identifiers."""
        return list(self._sources.keys())
