"""
QuantLab Reusable Pipeline Engine.

Allows building reusable processing pipelines, registering pipeline stages, validating inputs/outputs,
versioning pipeline definitions, and publishing stage telemetry events.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("PipelineEngine")


@dataclass
class PipelineStage:
    """Dataclass holding pipeline stage definition."""

    stage_name: str
    action_fn: Callable[[Any], Any]
    input_validator: Optional[Callable[[Any], bool]] = None
    output_validator: Optional[Callable[[Any], bool]] = None


class PipelineEngine:
    """Institutional Reusable Pipeline Engine."""

    def __init__(self, pipeline_name: str, version: str = "1.0.0", event_bus: Optional[StudioEventBus] = None) -> None:
        self.pipeline_name = pipeline_name
        self.version = version
        self.event_bus = event_bus or StudioEventBus()
        self.stages: List[PipelineStage] = []

    def add_stage(
        self,
        stage_name: str,
        action_fn: Callable[[Any], Any],
        input_validator: Optional[Callable[[Any], bool]] = None,
        output_validator: Optional[Callable[[Any], bool]] = None,
    ) -> "PipelineEngine":
        """Add processing stage to pipeline."""
        stage = PipelineStage(
            stage_name=stage_name,
            action_fn=action_fn,
            input_validator=input_validator,
            output_validator=output_validator,
        )
        self.stages.append(stage)
        return self

    def run_pipeline(self, initial_input: Any) -> Any:
        """Run pipeline data flow sequentially through all registered stages."""
        current_data = initial_input
        logger.info(f"Executing Pipeline '{self.pipeline_name}' (V{self.version})")

        for stage in self.stages:
            if stage.input_validator and not stage.input_validator(current_data):
                raise ValueError(f"Pipeline stage '{stage.stage_name}' input validation failed.")

            current_data = stage.action_fn(current_data)

            if stage.output_validator and not stage.output_validator(current_data):
                raise ValueError(f"Pipeline stage '{stage.stage_name}' output validation failed.")

        return current_data
