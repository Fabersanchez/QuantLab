"""
QuantLab Event Bus System.

Implements a decoupled publish-subscribe event bus for inter-component
communication within the quantitative engine.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List


@dataclass
class Event:
    """Represents a system event in QuantLab.

    Attributes:
        event_type: Unique identifier string for the event topic/type.
        payload: Optional data carried by the event.
        timestamp: Creation timestamp of the event.
    """

    event_type: str
    payload: Any = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


EventHandler = Callable[[Event], None]


class EventBus:
    """Decoupled Event Bus supporting pub/sub messaging pattern."""

    def __init__(self) -> None:
        """Initialize the EventBus with empty subscriber tables."""
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._broadcast_subscribers: List[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler callback to a specific event type.

        Args:
            event_type: The event type to listen for.
            handler: Callable callback taking an Event instance.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Unsubscribe a handler callback from a specific event type.

        Args:
            event_type: The event type topic.
            handler: The subscribed callback function.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        if (
            event_type in self._subscribers
            and handler in self._subscribers[event_type]
        ):
            self._subscribers[event_type].remove(handler)
            return True
        return False

    def publish(self, event_type_or_event: Any, payload: Any = None) -> int:
        """Publish an event to all subscribed handlers.

        Args:
            event_type_or_event: Event instance or event_type string.
            payload: Optional data payload (used if first argument is a string).

        Returns:
            Number of handlers notified.
        """
        if isinstance(event_type_or_event, Event):
            event = event_type_or_event
        else:
            event = Event(event_type=str(event_type_or_event), payload=payload)

        handlers = self._subscribers.get(event.event_type, [])
        notified = 0

        for handler in list(handlers):
            handler(event)
            notified += 1

        for broadcast_handler in list(self._broadcast_subscribers):
            broadcast_handler(event)
            notified += 1

        return notified

    def broadcast(self, payload: Any) -> int:
        """Broadcast payload to all broadcast subscribers as a BROADCAST event.

        Args:
            payload: Data payload to broadcast.

        Returns:
            Number of handlers notified.
        """
        return self.publish(event_type_or_event="BROADCAST", payload=payload)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to receive ALL published events."""
        if handler not in self._broadcast_subscribers:
            self._broadcast_subscribers.append(handler)

    def unsubscribe_all(self, handler: EventHandler) -> bool:
        """Unsubscribe a broadcast handler."""
        if handler in self._broadcast_subscribers:
            self._broadcast_subscribers.remove(handler)
            return True
        return False

    def clear(self) -> None:
        """Clear all event subscriptions."""
        self._subscribers.clear()
        self._broadcast_subscribers.clear()
