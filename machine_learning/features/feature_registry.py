"""
QuantLab Feature Registry.

Centralized repository for registering, tracking, querying, and auditing
predictive features and their metadata.
"""

from typing import Dict, List, Optional
from machine_learning.features.feature_metadata import FeatureMetadata


class FeatureAlreadyRegisteredError(Exception):
    """Raised when attempting to register a feature name that already exists."""

    pass


class FeatureNotFoundError(Exception):
    """Raised when looking up an unregistered feature."""

    pass


class FeatureRegistry:
    """Institutional registry for feature definitions and metadata."""

    def __init__(self) -> None:
        """Initialize an empty FeatureRegistry."""
        self._registry: Dict[str, FeatureMetadata] = {}

    def register(self, metadata: FeatureMetadata, overwrite: bool = False) -> None:
        """Register a feature metadata record.

        Args:
            metadata: FeatureMetadata object.
            overwrite: If True, overwrites existing registration.

        Raises:
            FeatureAlreadyRegisteredError: If feature name exists and overwrite is False.
        """
        if metadata.name in self._registry and not overwrite:
            raise FeatureAlreadyRegisteredError(
                f"Feature '{metadata.name}' is already registered."
            )
        self._registry[metadata.name] = metadata

    def unregister(self, name: str) -> FeatureMetadata:
        """Unregister and return a feature metadata record.

        Args:
            name: Identifier of feature to remove.

        Returns:
            The removed FeatureMetadata.

        Raises:
            FeatureNotFoundError: If feature is not registered.
        """
        if name not in self._registry:
            raise FeatureNotFoundError(f"Feature '{name}' not found.")
        return self._registry.pop(name)

    def get(self, name: str) -> FeatureMetadata:
        """Retrieve metadata for a registered feature by name.

        Args:
            name: Identifier of feature.

        Returns:
            FeatureMetadata object.

        Raises:
            FeatureNotFoundError: If feature is not registered.
        """
        if name not in self._registry:
            raise FeatureNotFoundError(f"Feature '{name}' not found.")
        return self._registry[name]

    def has(self, name: str) -> bool:
        """Return True if feature is registered, False otherwise."""
        return name in self._registry

    def search_by_category(self, category: str) -> List[FeatureMetadata]:
        """Search registered features matching a specific category string."""
        return [
            meta
            for meta in self._registry.values()
            if meta.category.lower() == category.lower()
        ]

    def list_features(self) -> List[str]:
        """Return list of all registered feature names."""
        return list(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered features."""
        self._registry.clear()
