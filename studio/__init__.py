"""
QuantLab Studio Enterprise Platform Package.

Centralizes, orchestrates, visualizes, and monitors all QuantLab quantitative scientific engines:
Application Shell, Navigation Router, Event Bus, Injected Services, Session Persistence,
Theme Engine, Notification Hub, System Monitoring, Dashboard Engine, Workspace Platform,
Project Manager, Project Explorer, Layout Engine, Widget Framework, Perspective Manager,
Session Crash Recovery Engine, State Synchronizer, Dataset Center, Data Quality Engine,
Experiment Center, Research Center, Artifact Manager, Metadata Platform, Data Lineage Engine,
Preview Engine, Catalog Engine, Job Manager, Workflow Engine, Task Engine, Execution Queue,
Resource Manager, Execution Supervisor, Worker Framework, Pipeline Engine, and Execution Center.
"""

from studio.artifact_manager import ArtifactManager
from studio.catalog import CatalogEngine
from studio.dashboard import (
    DashboardEngine,
    DashboardFoundation,
    DashboardWidgetCard,
    DashboardWidgetInstance,
)
from studio.dataset_center import DatasetCenter, DatasetRecord
from studio.events import (
    ModuleActivatedEvent,
    ModuleClosedEvent,
    NotificationCreatedEvent,
    ServiceConnectedEvent,
    ServiceDisconnectedEvent,
    StudioEvent,
    StudioEventBus,
    TaskFinishedEvent,
    TaskStartedEvent,
    ViewChangedEvent,
    WorkspaceClosedEvent,
    WorkspaceLoadedEvent,
)
from studio.execution_center import ExecutionCenter
from studio.experiment_center import ExperimentCenter, ExperimentRecord
from studio.explorer import ExplorerNode, ProjectExplorer
from studio.job_manager import JobManager, JobRecord
from studio.layouts import LayoutEngine, LayoutPanelConfig, StudioLayoutProfile
from studio.lineage import DataLineageEngine
from studio.logging import StudioLogger, get_studio_logger
from studio.metadata import EnrichedMetadata, MetadataPlatform
from studio.monitoring import MonitoringFramework, SystemTelemetrySnapshot
from studio.navigation import NavigationItem, NavigationRegistry
from studio.notifications import NotificationFramework, StudioNotification
from studio.perspectives import PerspectiveManager, StudioPerspective
from studio.pipeline import PipelineEngine, PipelineStage
from studio.preview import DatasetPreviewSummary, PreviewEngine
from studio.projects import EnterpriseProject, ProjectManager
from studio.queue import ExecutionQueue
from studio.research_center import ResearchCenter, ResearchLine
from studio.resource_manager import ResourceAllocationGrant, ResourceManager
from studio.services import (
    BaseService,
    ConfigurationService,
    DashboardService,
    MonitoringService,
    NavigationService,
    NotificationService,
    PluginService,
    ServiceContainer,
    SessionService,
    ThemeService,
    WorkspaceService,
)
from studio.sessions import CrashRecoveryCheckpoint, SessionRecoveryEngine
from studio.settings import SessionManager
from studio.shell import ApplicationShell, ShellState
from studio.studio_app import QuantLabStudioApp
from studio.supervisor import ExecutionSupervisor
from studio.sync import StateSynchronizer
from studio.task_engine import BaseTask, GenericTask, TaskEngine
from studio.themes import StudioThemeEngine, StudioThemePalette
from studio.validation import DataQualityEngine, QualityAlert
from studio.widgets import BaseWidget, GenericStudioWidget, WidgetFramework
from studio.workers import BaseWorker, GenericLocalWorker, WorkerFramework, WorkerInfo
from studio.workflow import WorkflowEngine, WorkflowStage
from studio.workspace import EnterpriseWorkspace, WorkspaceManager

__all__ = [
    "QuantLabStudioApp",
    "ApplicationShell",
    "ShellState",
    "StudioEventBus",
    "StudioEvent",
    "WorkspaceLoadedEvent",
    "WorkspaceClosedEvent",
    "ModuleActivatedEvent",
    "ModuleClosedEvent",
    "ViewChangedEvent",
    "NotificationCreatedEvent",
    "TaskStartedEvent",
    "TaskFinishedEvent",
    "ServiceConnectedEvent",
    "ServiceDisconnectedEvent",
    "ServiceContainer",
    "BaseService",
    "WorkspaceService",
    "NavigationService",
    "DashboardService",
    "NotificationService",
    "MonitoringService",
    "SessionService",
    "PluginService",
    "ThemeService",
    "ConfigurationService",
    "StudioThemeEngine",
    "StudioThemePalette",
    "NotificationFramework",
    "StudioNotification",
    "MonitoringFramework",
    "SystemTelemetrySnapshot",
    "SessionManager",
    "NavigationRegistry",
    "NavigationItem",
    "DashboardFoundation",
    "DashboardWidgetCard",
    "DashboardEngine",
    "DashboardWidgetInstance",
    "WorkspaceManager",
    "EnterpriseWorkspace",
    "ProjectManager",
    "EnterpriseProject",
    "ProjectExplorer",
    "ExplorerNode",
    "LayoutEngine",
    "LayoutPanelConfig",
    "StudioLayoutProfile",
    "BaseWidget",
    "GenericStudioWidget",
    "WidgetFramework",
    "PerspectiveManager",
    "StudioPerspective",
    "SessionRecoveryEngine",
    "CrashRecoveryCheckpoint",
    "StateSynchronizer",
    "DatasetCenter",
    "DatasetRecord",
    "DataQualityEngine",
    "QualityAlert",
    "ExperimentCenter",
    "ExperimentRecord",
    "ResearchCenter",
    "ResearchLine",
    "ArtifactManager",
    "MetadataPlatform",
    "EnrichedMetadata",
    "DataLineageEngine",
    "PreviewEngine",
    "DatasetPreviewSummary",
    "CatalogEngine",
    "JobManager",
    "JobRecord",
    "WorkflowEngine",
    "WorkflowStage",
    "BaseTask",
    "GenericTask",
    "TaskEngine",
    "ExecutionQueue",
    "ResourceManager",
    "ResourceAllocationGrant",
    "ExecutionSupervisor",
    "BaseWorker",
    "WorkerInfo",
    "GenericLocalWorker",
    "WorkerFramework",
    "PipelineEngine",
    "PipelineStage",
    "ExecutionCenter",
    "StudioLogger",
    "get_studio_logger",
]
