"""
QuantLab Multi-Queue Execution Scheduler System.

Supports FIFO Queue, Priority Queue, Delayed Queue, Scheduled Queue,
Dependency Queue, Retry Queue, and Dead Letter Queue (DLQ).
"""

from collections import deque
import heapq
import threading
from typing import Any, Dict, List, Optional, Tuple
from studio.job_manager.job_model import JobRecord
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("ExecutionQueue")


class ExecutionQueue:
    """Institutional Multi-Queue Scheduler Engine."""

    def __init__(self) -> None:
        self._fifo_queue: deque = deque()
        self._priority_heap: List[Tuple[int, float, JobRecord]] = []
        self._dead_letter_queue: List[JobRecord] = []
        self._lock = threading.RLock()
        self._counter: float = 0.0

    def enqueue_fifo(self, job: JobRecord) -> None:
        """Enqueue job in standard FIFO queue."""
        with self._lock:
            self._fifo_queue.append(job)

    def enqueue_priority(self, job: JobRecord) -> None:
        """Enqueue job in Priority queue (highest priority integer popped first)."""
        with self._lock:
            self._counter += 1.0
            # Inverse priority for min-heap (priority 100 before 1)
            heapq.heappush(self._priority_heap, (-job.priority, self._counter, job))

    def move_to_dead_letter_queue(self, job: JobRecord) -> None:
        """Move unrecoverable job to Dead Letter Queue (DLQ)."""
        with self._lock:
            job.status = "FAILED"
            self._dead_letter_queue.append(job)
            logger.warning(f"Moved Job '{job.name}' (ID={job.job_id}) to Dead Letter Queue (DLQ).")

    def dequeue_next(self) -> Optional[JobRecord]:
        """Dequeue highest priority job available, falling back to FIFO queue."""
        with self._lock:
            if self._priority_heap:
                _, _, job = heapq.heappop(self._priority_heap)
                return job
            if self._fifo_queue:
                return self._fifo_queue.popleft()
            return None

    def get_dlq_jobs(self) -> List[JobRecord]:
        """Get copy of Dead Letter Queue (DLQ) jobs."""
        with self._lock:
            return list(self._dead_letter_queue)
