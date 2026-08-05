"""
QuantLab Research Management Platform Engine.

Manages complete research lines, hypothesis tracking, collaborative research,
and scientific conclusions.
"""

from typing import Any, Dict, List, Optional
from studio.events.event_bus import StudioEventBus
from studio.logging.studio_logger import get_studio_logger
from studio.research_center.research_line import ResearchLine

logger = get_studio_logger("ResearchCenter")


class ResearchCenter:
    """Institutional Research Management Platform Engine."""

    def __init__(self, event_bus: Optional[StudioEventBus] = None) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self._research_lines: Dict[str, ResearchLine] = {}

    def create_research_line(
        self, title: str, hypothesis: str, author: str = "QuantResearcher"
    ) -> ResearchLine:
        """Create new quantitative research line."""
        line = ResearchLine(title=title, hypothesis=hypothesis, author=author)
        self._research_lines[line.line_id] = line
        logger.info(f"Created Research Line '{title}' (ID={line.line_id})")
        return line

    def add_experiment_to_line(self, line_id: str, experiment_id: str) -> bool:
        """Associate experiment run ID with target research line."""
        line = self._research_lines.get(line_id)
        if line and experiment_id not in line.experiments:
            line.experiments.append(experiment_id)
            return True
        return False

    def add_note_to_line(self, line_id: str, note: str) -> bool:
        """Add researcher note to target research line."""
        line = self._research_lines.get(line_id)
        if line:
            line.notes.append(note)
            return True
        return False

    def get_research_line(self, line_id: str) -> Optional[ResearchLine]:
        """Fetch ResearchLine by ID."""
        return self._research_lines.get(line_id)

    def list_research_lines(self) -> List[ResearchLine]:
        """List registered research lines."""
        return list(self._research_lines.values())
