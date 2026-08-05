"""
QuantLab Studio Abstract Base Service Specification.
"""

from abc import ABC, abstractmethod


class BaseService(ABC):
    """Abstract Base Class for all QuantLab Studio Injected Services."""

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize service resources."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown service resources."""
        pass
