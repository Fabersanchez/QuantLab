"""
QuantLab Task Engine Execution Manager.

Manages execution of atomic tasks enforcing standard lifecycle transitions and retry logic.
"""

from typing import Any, Dict, Optional
from studio.logging.studio_logger import get_studio_logger
from studio.task_engine.base_task import BaseTask

logger = get_studio_logger("TaskEngine")


class GenericTask(BaseTask):
    """Concrete implementation of BaseTask for standard execution routines."""

    def __init__(self, task_id: str, name: str, action_fn: Any) -> None:
        super().__init__(task_id, name)
        self.action_fn = action_fn

    def initialize(self) -> None:
        self.status = "INITIALIZED"

    def validate(self) -> bool:
        self.status = "VALIDATED"
        return True

    def schedule(self) -> None:
        self.status = "SCHEDULED"

    def execute(self) -> Dict[str, Any]:
        self.status = "EXECUTING"
        res = self.action_fn()
        return res if isinstance(res, dict) else {"result": res}

    def monitor(self) -> Dict[str, Any]:
        return {"status": self.status}

    def retry(self) -> bool:
        return True

    def complete(self) -> None:
        self.status = "COMPLETED"

    def rollback(self) -> None:
        self.status = "ROLLED_BACK"

    def destroy(self) -> None:
        self.status = "DESTROYED"


class TaskEngine:
    """Institutional Task Engine Execution Manager."""

    @staticmethod
    def run_task(task: BaseTask) -> Dict[str, Any]:
        """Run task through complete lifecycle."""
        task.initialize()
        if not task.validate():
            task.rollback()
            return {"error": "Validation failed"}

        task.schedule()
        try:
            res = task.execute()
            task.complete()
            logger.info(f"Task '{task.name}' executed successfully.")
            return res
        except Exception as e:
            logger.error(f"Task '{task.name}' failed: {e}")
            task.rollback()
            return {"error": str(e)}
        finally:
            task.destroy()
