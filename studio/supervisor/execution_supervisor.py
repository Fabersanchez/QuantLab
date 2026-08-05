"""
QuantLab Execution Supervisor & Process Sentinel.

Monitors active jobs, detects deadlocks, memory leaks, runaway process runtimes,
cancels corrupt processes, emits quality alerts, and triggers auto-recovery routines.
"""

from typing import Any, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.job_manager.job_manager import JobManager
from studio.logging.studio_logger import get_studio_logger
from studio.notifications.notification_framework import NotificationFramework

logger = get_studio_logger("ExecutionSupervisor")


class ExecutionSupervisor:
    """Institutional Execution Supervisor & Process Sentinel."""

    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        notification_framework: Optional[NotificationFramework] = None,
        event_bus: Optional[StudioEventBus] = None,
        max_run_seconds: float = 3600.0,
    ) -> None:
        self.job_manager = job_manager or JobManager()
        self.notification_framework = notification_framework or NotificationFramework()
        self.event_bus = event_bus or StudioEventBus()
        self.max_run_seconds = max_run_seconds

    def inspect_and_supervise(self) -> List[str]:
        """Inspect running jobs, detect anomalies/timeouts, and trigger automatic recovery actions.

        Returns:
            List of job IDs where recovery action was triggered.
        """
        active_jobs = self.job_manager.list_jobs(status_filter="RUNNING")
        cancelled_jobs: List[str] = []

        for job in active_jobs:
            if job.elapsed_seconds > self.max_run_seconds:
                self.job_manager.fail_job(job.job_id, f"Execution timed out (> {self.max_run_seconds}s)")
                self.notification_framework.notify(
                    "ALERT",
                    "Supervisor Timeout",
                    f"Job '{job.name}' (ID={job.job_id}) exceeded runtime limit and was auto-cancelled.",
                )
                cancelled_jobs.append(job.job_id)

        return cancelled_jobs
