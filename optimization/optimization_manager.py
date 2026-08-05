"""
QuantLab Master Optimization Manager.

Orchestrates multiple simultaneous optimization runs across CPU cores and memory bounds,
managing job queues, job priorities, retries, load balancing, resource limits, and cancellation.
"""

from dataclasses import dataclass, field
from queue import PriorityQueue
import threading
import time
from typing import Any, Dict, List, Optional
import psutil

from optimization.logger import get_optimization_logger
from optimization.optimizer import Optimizer

logger = get_optimization_logger("OptimizationManager")


@dataclass(order=True)
class OptimizationJob:
    """Dataclass holding a prioritized optimization job."""

    priority: int
    job_id: str = field(compare=False)
    optimizer: Optimizer = field(compare=False)
    max_evaluations: int = field(default=50, compare=False)
    batch_size: int = field(default=4, compare=False)
    retries: int = field(default=3, compare=False)
    status: str = field(default="QUEUED", compare=False)  # QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
    result: Optional[Any] = field(default=None, compare=False)
    error_message: str = field(default="", compare=False)


class OptimizationManager:
    """Institutional Multi-Job Optimization Manager."""

    def __init__(
        self,
        total_cpus: Optional[int] = None,
        max_memory_mb: Optional[float] = None,
    ) -> None:
        """Initialize OptimizationManager.

        Args:
            total_cpus: Total CPU cores allocated.
            max_memory_mb: Maximum memory budget MB.
        """
        self.total_cpus = total_cpus or (psutil.cpu_count(logical=True) or 4)
        mem = psutil.virtual_memory()
        self.max_memory_mb = max_memory_mb or (mem.total / (1024**2) * 0.8)

        self._job_queue: PriorityQueue = PriorityQueue()
        self._active_jobs: Dict[str, OptimizationJob] = {}
        self._completed_jobs: Dict[str, OptimizationJob] = {}
        self._lock = threading.RLock()
        self._job_counter: int = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running: bool = False

    def submit_job(
        self,
        optimizer: Optimizer,
        max_evaluations: int = 50,
        batch_size: int = 4,
        priority: int = 10,
        retries: int = 3,
    ) -> str:
        """Submit an optimization job to queue.

        Returns:
            Job ID string.
        """
        with self._lock:
            self._job_counter += 1
            job_id = f"JOB-{self._job_counter:04d}-{optimizer.opt_id}"
            job = OptimizationJob(
                priority=priority,
                job_id=job_id,
                optimizer=optimizer,
                max_evaluations=max_evaluations,
                batch_size=batch_size,
                retries=retries,
            )
            self._job_queue.put(job)
            self._active_jobs[job_id] = job
            logger.info(f"Submitted optimization job: ID={job_id} | Priority={priority}")

            if not self._is_running:
                self.start_manager()

            return job_id

    def start_manager(self) -> None:
        """Start background job processing worker loop."""
        with self._lock:
            if not self._is_running:
                self._is_running = True
                self._worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
                self._worker_thread.start()

    def _process_jobs(self) -> None:
        """Background worker thread processing queued optimization jobs."""
        while self._is_running:
            if self._job_queue.empty():
                time.sleep(0.2)
                continue

            try:
                job: OptimizationJob = self._job_queue.get(timeout=1.0)
            except Exception:
                continue

            job.status = "RUNNING"
            logger.info(f"Starting execution for job ID={job.job_id}")

            success = False
            attempts = 0
            while attempts < job.retries and not success:
                attempts += 1
                try:
                    res = job.optimizer.optimize(
                        max_evaluations=job.max_evaluations, batch_size=job.batch_size
                    )
                    job.result = res
                    job.status = "COMPLETED"
                    success = True
                    logger.info(f"Job completed successfully: ID={job.job_id}")
                except Exception as exc:
                    job.error_message = str(exc)
                    logger.error(f"Job {job.job_id} attempt {attempts} failed: {exc}")
                    if attempts < job.retries:
                        time.sleep(1.0)

            if not success:
                job.status = "FAILED"

            with self._lock:
                if job.job_id in self._active_jobs:
                    del self._active_jobs[job.job_id]
                self._completed_jobs[job.job_id] = job

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific optimization job."""
        with self._lock:
            if job_id in self._active_jobs:
                job = self._active_jobs[job_id]
                job.status = "CANCELLED"
                job.optimizer.cancel()
                del self._active_jobs[job_id]
                self._completed_jobs[job_id] = job
                return True
            return False

    def cancel_all(self) -> None:
        """Cancel all queued and active optimization jobs."""
        with self._lock:
            for job_id in list(self._active_jobs.keys()):
                self.cancel_job(job_id)
            self._is_running = False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a submitted job."""
        with self._lock:
            job = self._active_jobs.get(job_id) or self._completed_jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job.job_id,
                "status": job.status,
                "priority": job.priority,
                "retries": job.retries,
                "error_message": job.error_message,
                "has_result": job.result is not None,
            }
