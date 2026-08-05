"""
QuantLab Studio Typed Event Specifications.

Defines base StudioEvent dataclass and specialized events:
WorkspaceLoadedEvent, WorkspaceClosedEvent, ModuleActivatedEvent, ModuleClosedEvent,
ViewChangedEvent, NotificationCreatedEvent, TaskStartedEvent, TaskFinishedEvent,
ServiceConnectedEvent, ServiceDisconnectedEvent.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, Optional


@dataclass
class StudioEvent:
    """Base Event Class for QuantLab Studio Event-Driven Architecture."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "StudioEvent"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert event payload to dictionary."""
        return asdict(self)


@dataclass
class WorkspaceLoadedEvent(StudioEvent):
    """Fired when a workspace is opened or loaded."""

    workspace_id: str = ""
    workspace_name: str = ""
    path: str = ""

    def __post_init__(self) -> None:
        self.event_type = "WorkspaceLoaded"


@dataclass
class WorkspaceClosedEvent(StudioEvent):
    """Fired when active workspace is closed."""

    workspace_id: str = ""

    def __post_init__(self) -> None:
        self.event_type = "WorkspaceClosed"


@dataclass
class ModuleActivatedEvent(StudioEvent):
    """Fired when a Studio module view is activated."""

    module_id: str = ""
    module_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "ModuleActivated"


@dataclass
class ModuleClosedEvent(StudioEvent):
    """Fired when a Studio module view is closed."""

    module_id: str = ""

    def __post_init__(self) -> None:
        self.event_type = "ModuleClosed"


@dataclass
class ViewChangedEvent(StudioEvent):
    """Fired when active view layout or active tab changes."""

    old_view_id: str = ""
    new_view_id: str = ""

    def __post_init__(self) -> None:
        self.event_type = "ViewChanged"


@dataclass
class NotificationCreatedEvent(StudioEvent):
    """Fired when a new system notification is generated."""

    notification_id: str = ""
    severity: str = "INFO"
    title: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        self.event_type = "NotificationCreated"


@dataclass
class TaskStartedEvent(StudioEvent):
    """Fired when a background quantitative task starts."""

    task_id: str = ""
    task_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "TaskStarted"


@dataclass
class TaskFinishedEvent(StudioEvent):
    """Fired when a background quantitative task completes."""

    task_id: str = ""
    status: str = "SUCCESS"

    def __post_init__(self) -> None:
        self.event_type = "TaskFinished"


@dataclass
class ServiceConnectedEvent(StudioEvent):
    """Fired when an enterprise service connects."""

    service_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "ServiceConnected"


@dataclass
class ServiceDisconnectedEvent(StudioEvent):
    """Fired when an enterprise service disconnects."""

    service_name: str = ""

    def __post_init__(self) -> None:
        self.event_type = "ServiceDisconnected"
