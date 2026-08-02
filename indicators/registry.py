"""
QuantLab Indicator Registry.

Centralized institutional registry storing indicator classes, metadata,
parameters, dependencies, categories, and status.
"""

from typing import Dict, List, Optional, Type
from indicators.base_indicator import BaseIndicator
from indicators.metadata import IndicatorMetadata


class IndicatorAlreadyRegisteredError(Exception):
    """Raised when registering an indicator key that already exists."""

    pass


class IndicatorNotFoundError(Exception):
    """Raised when looking up an unregistered indicator."""

    pass


class IndicatorRegistry:
    """Central institutional registry for all quantitative indicators."""

    def __init__(self) -> None:
        """Initialize an empty IndicatorRegistry."""
        self._registry: Dict[str, Type[BaseIndicator]] = {}

    def register(
        self, indicator_cls: Type[BaseIndicator], overwrite: bool = False
    ) -> None:
        """Register an indicator class.

        Args:
            indicator_cls: Subclass of BaseIndicator.
            overwrite: If True, overwrites existing registration.

        Raises:
            IndicatorAlreadyRegisteredError: If indicator name already registered and overwrite is False.
        """
        meta = indicator_cls.metadata()
        name = meta.name.lower()
        if name in self._registry and not overwrite:
            raise IndicatorAlreadyRegisteredError(
                f"Indicator '{meta.name}' is already registered."
            )
        self._registry[name] = indicator_cls

    def unregister(self, name: str) -> Type[BaseIndicator]:
        """Remove and return an indicator class from registry.

        Args:
            name: Identifier string of indicator.

        Returns:
            The removed indicator class.

        Raises:
            IndicatorNotFoundError: If indicator is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise IndicatorNotFoundError(f"Indicator '{name}' not found.")
        return self._registry.pop(key)

    def get(self, name: str) -> Type[BaseIndicator]:
        """Retrieve an indicator class by name.

        Args:
            name: Indicator name string.

        Returns:
            Subclass of BaseIndicator.

        Raises:
            IndicatorNotFoundError: If name is not registered.
        """
        key = name.lower()
        if key not in self._registry:
            raise IndicatorNotFoundError(f"Indicator '{name}' not found in registry.")
        return self._registry[key]

    def has(self, name: str) -> bool:
        """Return True if indicator is registered, False otherwise."""
        return name.lower() in self._registry

    def search_by_category(self, category: str) -> List[IndicatorMetadata]:
        """Search registered indicators matching a specific category string."""
        return [
            cls.metadata()
            for cls in self._registry.values()
            if cls.metadata().category.lower() == category.lower()
        ]

    def list_indicators(self) -> List[str]:
        """Return list of registered indicator names."""
        return [cls.metadata().name for cls in self._registry.values()]

    def get_metadata(self, name: str) -> IndicatorMetadata:
        """Get IndicatorMetadata for a registered indicator by name."""
        return self.get(name).metadata()

    def clear(self) -> None:
        """Clear all registered indicators."""
        self._registry.clear()
