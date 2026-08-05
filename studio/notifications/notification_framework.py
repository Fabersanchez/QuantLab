"""
QuantLab Studio Notification Framework Engine.

Central notification hub supporting INFO, WARNING, ERROR, EVENT, PROCESS, ALERT notification types,
history querying, filtering by severity/source, and grouping.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional


@dataclass
class StudioNotification:
    """Dataclass holding notification item data."""

    notification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: str = "INFO"  # 'INFO', 'WARNING', 'ERROR', 'EVENT', 'PROCESS', 'ALERT'
    source_module: str = "Studio"
    title: str = "Notification"
    message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification item to dictionary."""
        return asdict(self)


class NotificationFramework:
    """Institutional Notification Framework Engine."""

    def __init__(self) -> None:
        self._history: List[StudioNotification] = []

    def notify(
        self, severity: str, title: str, message: str, source_module: str = "Studio"
    ) -> StudioNotification:
        """Create and register notification record."""
        note = StudioNotification(
            severity=severity.upper(),
            source_module=source_module,
            title=title,
            message=message,
        )
        self._history.append(note)
        return note

    def get_history(
        self, severity_filter: Optional[str] = None, source_filter: Optional[str] = None
    ) -> List[StudioNotification]:
        """Query notification history filtered by severity or source module."""
        filtered = self._history
        if severity_filter:
            filtered = [n for n in filtered if n.severity == severity_filter.upper()]
        if source_filter:
            filtered = [n for n in filtered if n.source_module.lower() == source_filter.lower()]
        return list(filtered)

    def clear(self) -> None:
        """Clear notification history."""
        self._history.clear()
