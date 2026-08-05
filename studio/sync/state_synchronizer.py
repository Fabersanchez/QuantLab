"""
QuantLab Studio Centralized State Synchronizer Engine.

Propagates all state mutations strictly via StudioEventBus with zero direct coupling between modules.
"""

from typing import Any, Dict, Optional
from studio.events.event_bus import StudioEventBus
from studio.events.studio_events import StudioEvent, ViewChangedEvent, WorkspaceLoadedEvent
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("StateSynchronizer")


class StateSynchronizer:
    """Institutional Centralized State Synchronizer Engine."""

    def __init__(self, event_bus: Optional[StudioEventBus] = None) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self._shared_state: Dict[str, Any] = {}
        self._event_history_count: int = 0

        # Subscribe to all events via event bus wildcard
        self.event_bus.subscribe("*", self._on_event_received)

    def _on_event_received(self, event: StudioEvent) -> None:
        """Handle incoming state mutation event from EventBus."""
        self._event_history_count += 1
        if isinstance(event, WorkspaceLoadedEvent):
            self._shared_state["active_workspace_id"] = event.workspace_id
            self._shared_state["active_workspace_name"] = event.workspace_name
        elif isinstance(event, ViewChangedEvent):
            self._shared_state["active_view_id"] = event.new_view_id

        logger.debug(f"StateSynchronizer synchronized event '{event.event_type}'")

    def get_state(self, key: str, default: Any = None) -> Any:
        """Fetch synchronized state key."""
        return self._shared_state.get(key, default)

    def set_state(self, key: str, value: Any, event_to_publish: Optional[StudioEvent] = None) -> None:
        """Mutate state key and publish corresponding event to EventBus."""
        self._shared_state[key] = value
        if event_to_publish:
            self.event_bus.publish(event_to_publish)
