"""
QuantLab Studio Master Enterprise Application Launcher.

Centralizes, initializes, and orchestrates:
- StudioEventBus
- ServiceContainer IoC
- ApplicationShell
- NavigationRegistry
- SessionManager
- StudioThemeEngine
- NotificationFramework
- MonitoringFramework
- DashboardFoundation
- WorkspaceManager
"""

from typing import Any, Dict, Optional

from studio.dashboard.dashboard_foundation import DashboardFoundation
from studio.events.event_bus import StudioEventBus
from studio.events.studio_events import ServiceConnectedEvent
from studio.logging.studio_logger import get_studio_logger
from studio.monitoring.monitoring_framework import MonitoringFramework
from studio.navigation.navigation_registry import NavigationRegistry
from studio.notifications.notification_framework import NotificationFramework
from studio.services.configuration_service import ConfigurationService
from studio.services.container import ServiceContainer
from studio.services.dashboard_service import DashboardService
from studio.services.monitoring_service import MonitoringService
from studio.services.navigation_service import NavigationService
from studio.services.notification_service import NotificationService
from studio.services.plugin_service import PluginService
from studio.services.session_service import SessionService
from studio.services.theme_service import ThemeService
from studio.services.workspace_service import WorkspaceService
from studio.settings.session_manager import SessionManager
from studio.shell.application_shell import ApplicationShell
from studio.themes.theme_engine import StudioThemeEngine
from studio.workspace.workspace_manager import WorkspaceManager

logger = get_studio_logger("StudioApp")


class QuantLabStudioApp:
    """Master QuantLab Studio Enterprise Application Launcher."""

    def __init__(self, session_filepath: str = "studio_session.json") -> None:
        """Initialize QuantLab Studio Application.

        Args:
            session_filepath: Session persistence JSON file path.
        """
        self.event_bus = StudioEventBus()
        self.session_manager = SessionManager(session_filepath=session_filepath)
        self.theme_engine = StudioThemeEngine()
        self.notification_framework = NotificationFramework()
        self.monitoring_framework = MonitoringFramework()
        self.dashboard_foundation = DashboardFoundation()
        self.navigation_registry = NavigationRegistry()
        self.workspace_manager = WorkspaceManager()

        self.container = ServiceContainer.get_instance()
        self._register_services()

        self.shell = ApplicationShell(
            event_bus=self.event_bus,
            navigation_registry=self.navigation_registry,
            session_manager=self.session_manager,
            theme_engine=self.theme_engine,
        )
        self._setup_default_routes()

    def _register_services(self) -> None:
        """Register all core enterprise services into IoC container."""
        self.container.register(WorkspaceService, WorkspaceService())
        self.container.register(NavigationService, NavigationService())
        self.container.register(DashboardService, DashboardService())
        self.container.register(NotificationService, NotificationService())
        self.container.register(MonitoringService, MonitoringService())
        self.container.register(SessionService, SessionService())
        self.container.register(PluginService, PluginService())
        self.container.register(ThemeService, ThemeService())
        self.container.register(ConfigurationService, ConfigurationService())

        self.event_bus.publish(ServiceConnectedEvent(service_name="ServiceContainer"))
        logger.info("All enterprise services registered into ServiceContainer.")

    def _setup_default_routes(self) -> None:
        """Register default core module navigation routes."""
        self.navigation_registry.register_module("dashboard", "Dashboard", icon="dashboard", order=10)
        self.navigation_registry.register_module("explorer", "Workspace Explorer", icon="folder", order=20)
        self.navigation_registry.register_module("experiments", "Experiment Center", icon="flask", order=30)
        self.navigation_registry.register_module("datasets", "Dataset Center", icon="database", order=40)
        self.navigation_registry.register_module("strategies", "Strategy Center", icon="chess", order=50)
        self.navigation_registry.register_module("optimization", "Optimization Center", icon="sliders", order=60)
        self.navigation_registry.register_module("portfolio", "Portfolio Center", icon="chart-pie", order=70)
        self.navigation_registry.register_module("registry", "Registry Governance", icon="award", order=80)
        self.navigation_registry.register_module("monitoring", "System Monitoring", icon="activity", order=90)
        self.navigation_registry.register_module("settings", "Settings", icon="settings", order=100)

    def run(self) -> Dict[str, Any]:
        """Launch Studio Application shell session.

        Returns:
            Studio app initialization status summary.
        """
        logger.info("QuantLab Studio Enterprise Application Started Successfully.")
        return {
            "status": "RUNNING",
            "active_theme": self.theme_engine.current_theme.name,
            "active_workspace": self.workspace_manager.active_workspace.name if self.workspace_manager.active_workspace else "None",
            "registered_modules": len(self.navigation_registry.list_modules()),
        }
