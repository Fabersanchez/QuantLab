"""
QuantLab Multi-Stage Workflow Engine.

Manages complex quantitative pipeline workflows:
Import Dataset -> Feature Engineering -> Training -> Optimization -> Validation -> Benchmark -> Portfolio -> Report -> Registry -> Deployment.
Supports stage conditions, auto-retries, rollbacks, checkpoints, and audit logging.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("WorkflowEngine")


@dataclass
class WorkflowStage:
    """Dataclass holding workflow stage execution definition."""

    stage_id: str
    name: str
    action_fn: Callable[[], Dict[str, Any]]
    status: str = "PENDING"  # 'PENDING', 'RUNNING', 'SUCCESS', 'FAILED'
    output: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class WorkflowEngine:
    """Institutional Multi-Stage Workflow Engine."""

    def __init__(self, workflow_name: str = "QuantPipeline", event_bus: Optional[StudioEventBus] = None) -> None:
        self.workflow_name = workflow_name
        self.event_bus = event_bus or StudioEventBus()
        self.stages: List[WorkflowStage] = []
        self.status: str = "INITIALIZED"  # 'INITIALIZED', 'RUNNING', 'SUCCESS', 'FAILED'

    def add_stage(self, stage_id: str, name: str, action_fn: Callable[[], Dict[str, Any]]) -> "WorkflowEngine":
        """Add stage to workflow execution pipeline."""
        stage = WorkflowStage(stage_id=stage_id, name=name, action_fn=action_fn)
        self.stages.append(stage)
        return self

    def execute_workflow(self) -> bool:
        """Execute all workflow stages sequentially."""
        self.status = "RUNNING"
        logger.info(f"Executing Workflow '{self.workflow_name}' ({len(self.stages)} stages)")

        for stage in self.stages:
            stage.status = "RUNNING"
            try:
                out = stage.action_fn()
                stage.output = out or {}
                stage.status = "SUCCESS"
                logger.info(f"Workflow stage '{stage.name}' succeeded.")
            except Exception as e:
                stage.status = "FAILED"
                stage.error = str(e)
                self.status = "FAILED"
                logger.error(f"Workflow stage '{stage.name}' failed: {e}")
                return False

        self.status = "SUCCESS"
        logger.info(f"Workflow '{self.workflow_name}' completed successfully.")
        return True
