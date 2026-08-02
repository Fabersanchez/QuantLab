"""
QuantLab Component Registry.

Provides a centralized, type-safe lookup repository for registering,
unregistering, querying, and managing system components and dependencies.
"""

from typing import Any, Dict, List, Optional


class ComponentAlreadyRegisteredError(Exception):
    """Raised when registering a component key that already exists."""

    pass


class ComponentNotFoundError(Exception):
    """Raised when looking up a component key that does not exist."""

    pass


class ComponentRegistry:
    """Central repository for system components and services.

    Prevents duplicate component registrations and provides search and query
    capabilities.
    """

    def __init__(self) -> None:
        """Initialize an empty ComponentRegistry."""
        self._components: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        component: Any,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a component under a unique string key.

        Args:
            name: Identifier for the component.
            component: Component instance or class to store.
            metadata: Optional dictionary containing metadata attributes.
            overwrite: If False, raises ComponentAlreadyRegisteredError on duplicate.

        Raises:
            ComponentAlreadyRegisteredError: If name already registered and overwrite is False.
        """
        if name in self._components and not overwrite:
            raise ComponentAlreadyRegisteredError(
                f"Component '{name}' is already registered."
            )
        self._components[name] = component
        self._metadata[name] = metadata or {}

    def unregister(self, name: str) -> Any:
        """Remove and return a registered component.

        Args:
            name: Identifier of the component to remove.

        Returns:
            The removed component instance.

        Raises:
            ComponentNotFoundError: If name is not registered.
        """
        if name not in self._components:
            raise ComponentNotFoundError(f"Component '{name}' not found.")
        self._metadata.pop(name, None)
        return self._components.pop(name)

    def get(self, name: str) -> Any:
        """Retrieve a component by name.

        Args:
            name: Identifier of the component.

        Returns:
            The registered component.

        Raises:
            ComponentNotFoundError: If name is not registered.
        """
        if name not in self._components:
            raise ComponentNotFoundError(f"Component '{name}' not found.")
        return self._components[name]

    def has(self, name: str) -> bool:
        """Return True if a component is registered under name, False otherwise."""
        return name in self._components

    def search(self, key_prefix: str) -> Dict[str, Any]:
        """Search registered components whose name starts with key_prefix.

        Args:
            key_prefix: Prefix string to filter component names.

        Returns:
            Dictionary of matching component names and instances.
        """
        return {
            k: v for k, v in self._components.items() if k.startswith(key_prefix)
        }

    def list_components(self) -> List[str]:
        """Return a list of all registered component identifiers."""
        return list(self._components.keys())

    def clear(self) -> None:
        """Remove all registered components."""
        self._components.clear()
        self._metadata.clear()
