"""
QuantLab Base Widget Interface Specification.

Defines standard institutional lifecycle methods for all Studio widgets:
Initialize(), Load(), Activate(), Refresh(), Suspend(), Resume(), Destroy().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseWidget(ABC):
    """Abstract Base Class for QuantLab Studio UI Widgets."""

    def __init__(self, widget_id: str, title: str) -> None:
        self.widget_id = widget_id
        self.title = title
        self.is_active: bool = False
        self.is_loaded: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize widget resources."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load data payloads into widget."""
        pass

    @abstractmethod
    def activate(self) -> None:
        """Activate widget view state."""
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Refresh real-time widget metrics."""
        pass

    @abstractmethod
    def suspend(self) -> None:
        """Suspend background updates when tab is hidden."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Resume background updates when tab becomes visible."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Clean up and release all widget resources."""
        pass
