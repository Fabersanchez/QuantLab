"""
QuantLab Base Worker Specification.

Defines standard institutional interface for execution workers:
Register(), Describe(), AcceptTask(), RejectTask(), ReportMetrics(), Pause(), Resume(), Stop(), Recover().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WorkerInfo:
    """Dataclass holding worker capability metadata."""

    worker_id: str
    name: str
    host: str = "localhost"
    max_concurrent_tasks: int = 4
    active_tasks_count: int = 0
    status: str = "IDLE"  # 'IDLE', 'BUSY', 'PAUSED', 'STOPPED'


class BaseWorker(ABC):
    """Abstract Base Class for Execution Workers."""

    def __init__(self, worker_id: str, name: str) -> None:
        self.info = WorkerInfo(worker_id=worker_id, name=name)

    @abstractmethod
    def accept_task(self, task_id: str) -> bool:
        """Accept task execution assignment."""
        pass

    @abstractmethod
    def report_metrics(self) -> Dict[str, Any]:
        """Report worker performance metrics."""
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pause worker task processing."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Resume worker task processing."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop worker engine."""
        pass
