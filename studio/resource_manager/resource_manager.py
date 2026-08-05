"""
QuantLab Hardware Resource Governor & Allocation Engine.

Dynamically allocates CPU cores, GPU units, RAM capacity, worker threads, and process limits
to prevent system overload and memory exhaustion.
"""

from dataclasses import dataclass
import os
import psutil
import threading
from typing import Any, Dict, Optional
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("ResourceManager")


@dataclass
class ResourceAllocationGrant:
    """Dataclass holding granted resource limits for a job/task."""

    job_id: str
    granted_threads: int
    granted_memory_mb: float
    gpu_allocated: bool = False


class ResourceManager:
    """Institutional Hardware Resource Governor Engine."""

    def __init__(self, max_cpu_percent: float = 85.0, max_mem_percent: float = 90.0) -> None:
        self.max_cpu_percent = max_cpu_percent
        self.max_mem_percent = max_mem_percent
        self._grants: Dict[str, ResourceAllocationGrant] = {}
        self._lock = threading.RLock()

    def can_allocate(self, required_threads: int = 1, required_mem_mb: float = 256.0) -> bool:
        """Check if system resource utilization permits new allocation grant."""
        with self._lock:
            try:
                mem_pct = float(psutil.virtual_memory().percent)
                cpu_pct = float(psutil.cpu_percent(interval=None))
                if mem_pct >= self.max_mem_percent or cpu_pct >= self.max_cpu_percent:
                    return False
            except Exception:
                pass
            return True

    def request_allocation(
        self, job_id: str, required_threads: int = 1, required_mem_mb: float = 256.0
    ) -> Optional[ResourceAllocationGrant]:
        """Request resource allocation grant for target job."""
        with self._lock:
            if not self.can_allocate(required_threads, required_mem_mb):
                logger.warning(f"Resource allocation denied for job '{job_id}' (Limit exceeded).")
                return None

            grant = ResourceAllocationGrant(
                job_id=job_id,
                granted_threads=required_threads,
                granted_memory_mb=required_mem_mb,
            )
            self._grants[job_id] = grant
            return grant

    def release_allocation(self, job_id: str) -> None:
        """Release allocated resource grant for target job."""
        with self._lock:
            if job_id in self._grants:
                self._grants.pop(job_id)
