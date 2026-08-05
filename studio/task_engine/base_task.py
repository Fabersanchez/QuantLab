"""
QuantLab Base Task Interface Specification.

Defines standard institutional lifecycle methods for atomic tasks:
Initialize(), Validate(), Schedule(), Execute(), Monitor(), Retry(), Complete(), Rollback(), Destroy().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTask(ABC):
    """Abstract Base Class for QuantLab Studio Executable Tasks."""

    def __init__(self, task_id: str, name: str) -> None:
        self.task_id = task_id
        self.name = name
        self.status: str = "INITIALIZED"  # 'INITIALIZED', 'VALIDATED', 'SCHEDULED', 'EXECUTING', 'COMPLETED', 'FAILED'

    @abstractmethod
    def initialize(self) -> None:
        """Initialize task resources."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate task inputs and prerequisites."""
        pass

    @abstractmethod
    def schedule(self) -> None:
        """Mark task as scheduled for execution."""
        pass

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute core task logic."""
        pass

    @abstractmethod
    def monitor(self) -> Dict[str, Any]:
        """Fetch task monitoring metrics."""
        pass

    @abstractmethod
    def retry(self) -> bool:
        """Execute retry policy after failure."""
        pass

    @abstractmethod
    def complete(self) -> None:
        """Complete task execution and release locks."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback task side effects after error."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """Clean up all task resources."""
        pass
