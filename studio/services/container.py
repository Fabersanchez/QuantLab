"""
QuantLab Studio Dependency Injection (IoC) Service Container.

Registers and resolves singleton and factory services across QuantLab Studio.
"""

import threading
from typing import Any, Dict, Optional, Type, TypeVar
from studio.services.base_service import BaseService

T = TypeVar("T", bound=BaseService)


class ServiceContainer:
    """Institutional Dependency Injection Service Container."""

    _instance: Optional["ServiceContainer"] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._services: Dict[Type[Any], BaseService] = {}
        self._named_services: Dict[str, BaseService] = {}

    @classmethod
    def get_instance(cls) -> "ServiceContainer":
        """Get singleton ServiceContainer instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, service_type: Type[T], instance: T) -> None:
        """Register service instance under service interface type."""
        with self._lock:
            self._services[service_type] = instance
            self._named_services[instance.service_name] = instance
            if not instance.is_initialized:
                instance.initialize()

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve registered service instance by type.

        Raises:
            KeyError: If service is not registered.
        """
        with self._lock:
            if service_type not in self._services:
                raise KeyError(f"Service of type '{service_type.__name__}' is not registered in container.")
            return self._services[service_type]  # type: ignore

    def resolve_by_name(self, name: str) -> Optional[BaseService]:
        """Resolve registered service by name."""
        with self._lock:
            return self._named_services.get(name)

    def clear(self) -> None:
        """Shutdown all services and clear container."""
        with self._lock:
            for s in self._services.values():
                try:
                    s.shutdown()
                except Exception:
                    pass
            self._services.clear()
            self._named_services.clear()
