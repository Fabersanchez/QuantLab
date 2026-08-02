"""
QuantLab Strategy Registry.

Centralized institutional registry for storing, querying, versioning,
enabling, disabling, and managing strategy definitions.
"""

from typing import Dict, List, Optional, Type
from strategies.base_strategy import BaseStrategy
from strategies.strategy_metadata import StrategyMetadata


class StrategyAlreadyRegisteredError(Exception):
    """Raised when registering a strategy key that already exists."""

    pass


class StrategyNotFoundError(Exception):
    """Raised when looking up an unregistered strategy."""

    pass


class StrategyRegistry:
    """Central institutional registry for quantitative strategies."""

    def __init__(self) -> None:
        """Initialize an empty StrategyRegistry."""
        self._registry: Dict[str, Type[BaseStrategy]] = {}
        self._enabled_status: Dict[str, bool] = {}

    def register(
        self, strategy_cls: Type[BaseStrategy], overwrite: bool = False
    ) -> None:
        """Register a strategy class.

        Args:
            strategy_cls: Subclass of BaseStrategy.
            overwrite: If True, overwrites existing registration.

        Raises:
            StrategyAlreadyRegisteredError: If strategy name already registered and overwrite is False.
        """
        meta = strategy_cls.metadata()
        key = meta.name.lower()
        if key in self._registry and not overwrite:
            raise StrategyAlreadyRegisteredError(
                f"Strategy '{meta.name}' is already registered."
            )
        self._registry[key] = strategy_cls
        self._enabled_status[key] = True

    def unregister(self, name: str) -> Type[BaseStrategy]:
        """Remove and return a strategy class from registry."""
        key = name.lower()
        if key not in self._registry:
            raise StrategyNotFoundError(f"Strategy '{name}' not found.")
        self._enabled_status.pop(key, None)
        return self._registry.pop(key)

    def get(self, name: str) -> Type[BaseStrategy]:
        """Retrieve a strategy class by name."""
        key = name.lower()
        if key not in self._registry:
            raise StrategyNotFoundError(f"Strategy '{name}' not found.")
        return self._registry[key]

    def has(self, name: str) -> bool:
        """Return True if strategy is registered, False otherwise."""
        return name.lower() in self._registry

    def enable(self, name: str) -> None:
        """Enable execution of a strategy by name."""
        key = name.lower()
        if key not in self._registry:
            raise StrategyNotFoundError(f"Strategy '{name}' not found.")
        self._enabled_status[key] = True

    def disable(self, name: str) -> None:
        """Disable execution of a strategy by name."""
        key = name.lower()
        if key not in self._registry:
            raise StrategyNotFoundError(f"Strategy '{name}' not found.")
        self._enabled_status[key] = False

    def is_enabled(self, name: str) -> bool:
        """Return True if strategy is enabled, False otherwise."""
        key = name.lower()
        return self._enabled_status.get(key, False)

    def search_by_category(self, category: str) -> List[StrategyMetadata]:
        """Search registered strategies matching a specific category."""
        return [
            cls.metadata()
            for cls in self._registry.values()
            if cls.metadata().category.lower() == category.lower()
        ]

    def list_strategies(self) -> List[str]:
        """Return list of all registered strategy names."""
        return [cls.metadata().name for cls in self._registry.values()]

    def list_enabled_strategies(self) -> List[str]:
        """Return list of currently enabled strategy names."""
        return [
            cls.metadata().name
            for key, cls in self._registry.items()
            if self._enabled_status.get(key, False)
        ]

    def clear(self) -> None:
        """Clear all registered strategies."""
        self._registry.clear()
        self._enabled_status.clear()
