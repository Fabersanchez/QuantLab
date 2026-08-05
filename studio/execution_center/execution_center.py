"""
QuantLab Studio Execution Center Control Hub Engine.

Orchestrates real-time active/pending/completed jobs telemetry, worker pool status,
ETA calculations, resource governor utilization, and action controls (pause, resume, cancel, retry, prioritize).
"""

from typing import Any, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.job_manager.job_manager import JobManager
from studio.job_manager.job_model import JobRecord
from studio.queue.execution_queue import ExecutionQueue
from studio.resource_manager.resource_manager import ResourceManager
from studio.supervisor.execution_supervisor import ExecutionSupervisor
from studio.workers.worker_framework import WorkerFramework


class ExecutionCenter:
    """Institutional Studio Execution Center Control Hub Engine."""

    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        queue: Optional[ExecutionQueue] = None,
        resource_manager: Optional[ResourceManager] = None,
        supervisor: Optional[ExecutionSupervisor] = None,
        worker_framework: Optional[WorkerFramework] = None,
        event_bus: Optional[StudioEventBus] = None,
    ) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self.job_manager = job_manager or JobManager(event_bus=self.event_bus)
        self.queue = queue or ExecutionQueue()
        self.resource_manager = resource_manager or ResourceManager()
        self.supervisor = supervisor or ExecutionSupervisor(job_manager=self.job_manager, event_bus=self.event_bus)
        self.worker_framework = worker_framework or WorkerFramework()

    def get_execution_telemetry(self) -> Dict[str, Any]:
        """Fetch comprehensive real-time execution telemetry dashboard metrics.

        Returns:
            Dictionary containing active, pending, completed jobs, DLQ count, and worker pool summary.
        """
        all_jobs = self.job_manager.list_jobs()
        active = [j for j in all_jobs if j.status == "RUNNING"]
        pending = [j for j in all_jobs if j.status == "PENDING"]
        completed = [j for j in all_jobs if j.status in ("SUCCESS", "FAILED")]
        dlq = self.queue.get_dlq_jobs()
        workers = self.worker_framework.list_workers()

        return {
            "active_jobs_count": len(active),
            "pending_jobs_count": len(pending),
            "completed_jobs_count": len(completed),
            "dead_letter_queue_count": len(dlq),
            "workers_count": len(workers),
            "can_allocate_resources": self.resource_manager.can_allocate(),
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel execution of target job."""
        return self.job_manager.cancel_job(job_id)

    def prioritize_job(self, job_id: str, new_priority: int) -> bool:
        """Update job priority level."""
        job = self.job_manager.get_job(job_id)
        if job:
            job.priority = new_priority
            return True
        return False
