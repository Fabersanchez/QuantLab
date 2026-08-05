"""
QuantLab Enterprise Job Manager Engine.

Coordinates job registration, dependency graph resolution, priority scheduling, state transitions
(PENDING, RUNNING, PAUSED, CANCELLED, SUCCESS, FAILED), and execution logging.
"""

from datetime import datetime, timezone
import threading
from typing import Any, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.events.studio_events import TaskFinishedEvent, TaskStartedEvent
from studio.job_manager.job_model import JobRecord
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("JobManager")


class JobManager:
    """Institutional Enterprise Job Manager Engine."""

    def __init__(self, event_bus: Optional[StudioEventBus] = None) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()

    def create_job(
        self,
        name: str,
        job_type: str = "Backtest",
        priority: int = 50,
        dependencies: Optional[List[str]] = None,
        estimated_seconds: float = 60.0,
        user: str = "QuantResearcher",
        workspace_id: str = "default_workspace",
        project_id: str = "",
    ) -> JobRecord:
        """Create and register new quantitative JobRecord."""
        with self._lock:
            job = JobRecord(
                name=name,
                job_type=job_type,
                priority=priority,
                dependencies=dependencies or [],
                estimated_seconds=estimated_seconds,
                user=user,
                workspace_id=workspace_id,
                project_id=project_id,
            )
            self._jobs[job.job_id] = job
            logger.info(f"Created Enterprise Job '{name}' (ID={job.job_id}, Type={job_type}, Priority={priority})")
            return job

    def start_job(self, job_id: str) -> bool:
        """Transition target job state to RUNNING."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ("PENDING", "PAUSED"):
                job.status = "RUNNING"
                job.start_time = datetime.now(timezone.utc).isoformat()
                self.event_bus.publish(TaskStartedEvent(task_id=job.job_id, task_name=job.name))
                logger.info(f"Started Job '{job.name}' (ID={job_id})")
                return True
            return False

    def complete_job(self, job_id: str, result_payload: Optional[Dict[str, Any]] = None) -> bool:
        """Transition target job state to SUCCESS."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == "RUNNING":
                job.status = "SUCCESS"
                job.progress_percent = 100.0
                job.end_time = datetime.now(timezone.utc).isoformat()
                if result_payload:
                    job.result_payload = result_payload
                self.event_bus.publish(TaskFinishedEvent(task_id=job.job_id, status="SUCCESS"))
                logger.info(f"Completed Job '{job.name}' (ID={job_id}) successfully.")
                return True
            return False

    def fail_job(self, job_id: str, error_msg: str) -> bool:
        """Transition target job state to FAILED."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "FAILED"
                job.end_time = datetime.now(timezone.utc).isoformat()
                job.errors.append(error_msg)
                self.event_bus.publish(TaskFinishedEvent(task_id=job.job_id, status="FAILED"))
                logger.error(f"Job '{job.name}' (ID={job_id}) failed: {error_msg}")
                return True
            return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel target pending/running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ("PENDING", "RUNNING", "PAUSED"):
                job.status = "CANCELLED"
                job.end_time = datetime.now(timezone.utc).isoformat()
                logger.info(f"Cancelled Job '{job.name}' (ID={job_id})")
                return True
            return False

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch JobRecord by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, status_filter: Optional[str] = None) -> List[JobRecord]:
        """List registered jobs filtered by status ordered by priority descending."""
        with self._lock:
            jobs = list(self._jobs.values())
            if status_filter:
                jobs = [j for j in jobs if j.status == status_filter.upper()]
            jobs.sort(key=lambda j: j.priority, reverse=True)
            return jobs
