"""
QuantLab Studio Event Bus Engine.

Thread-safe, decoupled event-driven pub/sub event bus supporting typed handlers,
topic subscriptions, asynchronous dispatching, and subscriber error isolation.
"""

from collections import defaultdict
import threading
from typing import Any, Callable, Dict, List, Type, Union
from studio.events.studio_events import StudioEvent
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("EventBus")


class StudioEventBus:
    """Institutional Thread-Safe Event Bus Engine."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[StudioEvent], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_history: List[StudioEvent] = []

    def subscribe(self, event_type: Union[str, Type[StudioEvent]], handler: Callable[[Any], None]) -> None:
        """Subscribe handler function to event type.

        Args:
            event_type: Event type string or StudioEvent class type.
            handler: Callable callback receiving event instance.
        """
        key = event_type if isinstance(event_type, str) else event_type.__name__
        with self._lock:
            if handler not in self._subscribers[key]:
                self._subscribers[key].append(handler)

    def unsubscribe(self, event_type: Union[str, Type[StudioEvent]], handler: Callable[[Any], None]) -> None:
        """Unsubscribe handler function from event type."""
        key = event_type if isinstance(event_type, str) else event_type.__name__
        with self._lock:
            if handler in self._subscribers[key]:
                self._subscribers[key].remove(handler)

    def publish(self, event: StudioEvent) -> None:
        """Publish event to all registered subscriber callbacks."""
        with self._lock:
            self._event_history.append(event)

        event_name = event.__class__.__name__
        key_str = event.event_type

        handlers_to_call = []
        with self._lock:
            handlers_to_call.extend(self._subscribers.get(event_name, []))
            if key_str != event_name:
                handlers_to_call.extend(self._subscribers.get(key_str, []))
            handlers_to_call.extend(self._subscribers.get("*", []))

        for handler in handlers_to_call:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error handling event '{event_name}' in handler {handler}: {e}")

    def clear(self) -> None:
        """Reset event bus subscriptions and history."""
        with self._lock:
            self._subscribers.clear()
            self._event_history.clear()

    @property
    def event_history(self) -> List[StudioEvent]:
        """Get copy of published event history."""
        with self._lock:
            return list(self._event_history)
