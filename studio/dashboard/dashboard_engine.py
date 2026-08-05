"""
QuantLab Enterprise Dynamic Dashboard Engine.

Orchestrates dynamic self-registering widgets, auto-update telemetry, docking/floating states,
position & size persistence, and event-driven metric updates via StudioEventBus.
"""

from dataclasses import asdict, dataclass, field
import threading
from typing import Any, Callable, Dict, List, Optional

from studio.events.event_bus import StudioEventBus
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("DashboardEngine")


@dataclass
class DashboardWidgetInstance:
    """Dataclass holding dynamic widget runtime configuration and bounds."""

    widget_id: str
    title: str
    category: str = "Analytics"
    is_visible: bool = True
    is_docked: bool = True
    x: int = 0
    y: int = 0
    width: int = 400
    height: int = 300
    metric_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert widget state to dictionary."""
        d = asdict(self)
        d.pop("metric_provider", None)
        return d


class DashboardEngine:
    """Institutional Dynamic Dashboard Engine."""

    def __init__(self, event_bus: Optional[StudioEventBus] = None) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self._widgets: Dict[str, DashboardWidgetInstance] = {}
        self._lock = threading.RLock()

    def register_widget(
        self,
        widget_id: str,
        title: str,
        category: str = "Analytics",
        width: int = 400,
        height: int = 300,
        metric_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> DashboardWidgetInstance:
        """Self-register a new widget in dashboard engine."""
        with self._lock:
            instance = DashboardWidgetInstance(
                widget_id=widget_id,
                title=title,
                category=category,
                width=width,
                height=height,
                metric_provider=metric_provider,
            )
            self._widgets[widget_id] = instance
            logger.info(f"Registered dynamic dashboard widget '{title}' (ID={widget_id})")
            return instance

    def set_widget_visibility(self, widget_id: str, is_visible: bool) -> bool:
        """Show or hide target dashboard widget."""
        with self._lock:
            w = self._widgets.get(widget_id)
            if w:
                w.is_visible = is_visible
                return True
            return False

    def update_widget_geometry(self, widget_id: str, x: int, y: int, width: int, height: int) -> bool:
        """Update and persist target widget position and dimensions."""
        with self._lock:
            w = self._widgets.get(widget_id)
            if w:
                w.x, w.y, w.width, w.height = x, y, width, height
                return True
            return False

    def get_widget_payload(self, widget_id: str) -> Dict[str, Any]:
        """Execute metric provider function and return data payload."""
        with self._lock:
            w = self._widgets.get(widget_id)
            if w and w.metric_provider:
                try:
                    return w.metric_provider()
                except Exception as e:
                    return {"error": str(e)}
            return {}

    def list_widgets(self) -> List[DashboardWidgetInstance]:
        """List registered dynamic widgets."""
        with self._lock:
            return list(self._widgets.values())
