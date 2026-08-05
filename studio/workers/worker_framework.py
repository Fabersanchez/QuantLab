"""
QuantLab Worker Pool Framework Engine.

Manages local and remote worker registration, capability self-description, task allocation,
load balancing, and worker state management.
"""

from typing import Any, Dict, List, Optional
from studio.logging.studio_logger import get_studio_logger
from studio.workers.base_worker import BaseWorker, WorkerInfo

logger = get_studio_logger("WorkerFramework")


class GenericLocalWorker(BaseWorker):
    """Concrete implementation of BaseWorker for local execution worker threads."""

    def accept_task(self, task_id: str) -> bool:
        if self.info.active_tasks_count < self.info.max_concurrent_tasks:
            self.info.active_tasks_count += 1
            self.info.status = "BUSY"
            return True
        return False

    def report_metrics(self) -> Dict[str, Any]:
        return {
            "worker_id": self.info.worker_id,
            "status": self.info.status,
            "active_tasks": self.info.active_tasks_count,
        }

    def pause(self) -> None:
        self.info.status = "PAUSED"

    def resume(self) -> None:
        self.info.status = "IDLE" if self.info.active_tasks_count == 0 else "BUSY"

    def stop(self) -> None:
        self.info.status = "STOPPED"


class WorkerFramework:
    """Institutional Worker Pool Framework Engine."""

    def __init__(self) -> None:
        self._workers: Dict[str, BaseWorker] = {}

    def register_worker(self, worker: BaseWorker) -> None:
        """Register worker instance into pool."""
        self._workers[worker.info.worker_id] = worker
        logger.info(f"Registered Worker '{worker.info.name}' (ID={worker.info.worker_id})")

    def find_available_worker(self) -> Optional[BaseWorker]:
        """Find idle or available worker in pool."""
        for w in self._workers.values():
            if w.info.status in ("IDLE", "BUSY") and w.info.active_tasks_count < w.info.max_concurrent_tasks:
                return w
        return None

    def list_workers(self) -> List[BaseWorker]:
        """List registered worker instances."""
        return list(self._workers.values())
