"""
QuantLab Module Manager.

Provides dynamic registration, loading, unloading, and lifecycle tracking
for decoupled QuantLab extensions and subsystems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseModule(ABC):
    """Abstract Base Class for all QuantLab modular extensions.

    Subclasses must implement name property, initialize, and shutdown methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique module name identifier."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize module resources, event listeners, and services."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up module resources and detach listeners."""
        pass


class ModuleNotFoundError(Exception):
    """Raised when requesting an unregistered or unloaded module."""

    pass


class ModuleAlreadyRegisteredError(Exception):
    """Raised when registering a module with an existing name."""

    pass


class ModuleManager:
    """Manages registration, lifecycle loading, and access to system modules."""

    def __init__(self) -> None:
        """Initialize ModuleManager with empty module tables."""
        self._registered_modules: Dict[str, BaseModule] = {}
        self._loaded_modules: Dict[str, BaseModule] = {}

    def register_module(self, module: BaseModule) -> None:
        """Register a module instance.

        Args:
            module: Instance implementing BaseModule interface.

        Raises:
            ModuleAlreadyRegisteredError: If module.name is already registered.
        """
        if module.name in self._registered_modules:
            raise ModuleAlreadyRegisteredError(
                f"Module '{module.name}' is already registered."
            )
        self._registered_modules[module.name] = module

    def load_module(self, name: str) -> BaseModule:
        """Load and initialize a registered module by name.

        Args:
            name: Unique name of the registered module.

        Returns:
            The loaded BaseModule instance.

        Raises:
            ModuleNotFoundError: If module is not registered.
        """
        if name not in self._registered_modules:
            raise ModuleNotFoundError(f"Module '{name}' is not registered.")

        module = self._registered_modules[name]
        if name not in self._loaded_modules:
            module.initialize()
            self._loaded_modules[name] = module
        return module

    def unload_module(self, name: str) -> None:
        """Unload and shut down an active module by name.

        Args:
            name: Unique name of the active module.

        Raises:
            ModuleNotFoundError: If module is not active or not registered.
        """
        if name not in self._loaded_modules:
            raise ModuleNotFoundError(f"Module '{name}' is not currently loaded.")

        module = self._loaded_modules.pop(name)
        module.shutdown()

    def get_module(self, name: str) -> BaseModule:
        """Retrieve a loaded module by name.

        Args:
            name: Unique name of the loaded module.

        Returns:
            The loaded BaseModule instance.

        Raises:
            ModuleNotFoundError: If module is not loaded.
        """
        if name not in self._loaded_modules:
            raise ModuleNotFoundError(f"Module '{name}' is not currently loaded.")
        return self._loaded_modules[name]

    def list_modules(self) -> List[str]:
        """Return a list of all registered module names."""
        return list(self._registered_modules.keys())

    def list_loaded_modules(self) -> List[str]:
        """Return a list of currently active/loaded module names."""
        return list(self._loaded_modules.keys())
