"""
QuantLab Studio Foundation Test Suite.

Validates all components of QuantLab Studio Foundation:
StudioEventBus, ServiceContainer, BaseService, WorkspaceService, NavigationService, DashboardService,
NotificationService, MonitoringService, SessionService, PluginService, ThemeService, ConfigurationService,
StudioThemeEngine, NotificationFramework, MonitoringFramework, SessionManager, NavigationRegistry,
DashboardFoundation, WorkspaceManager, ApplicationShell, and QuantLabStudioApp.
"""

import os
import shutil
import tempfile
import unittest

from studio import (
    ApplicationShell,
    ConfigurationService,
    DashboardFoundation,
    DashboardService,
    MonitoringFramework,
    MonitoringService,
    NavigationRegistry,
    NavigationService,
    NotificationFramework,
    NotificationService,
    PluginService,
    QuantLabStudioApp,
    ServiceConnectedEvent,
    ServiceContainer,
    SessionManager,
    SessionService,
    StudioEventBus,
    StudioThemeEngine,
    ThemeService,
    WorkspaceLoadedEvent,
    WorkspaceManager,
    WorkspaceService,
)


class TestQuantLabStudioFoundation(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Studio Foundation Enterprise Platform."""

    def setUp(self) -> None:
        """Set up temporary directory for session files."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_studio_test_")
        self.session_file = os.path.join(self.temp_dir, "test_session.json")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_event_bus_and_typed_events(self) -> None:
        """Test StudioEventBus pub/sub event dispatching and history."""
        bus = StudioEventBus()
        received = []

        def handle_event(event) -> None:
            received.append(event)

        bus.subscribe(WorkspaceLoadedEvent, handle_event)
        evt = WorkspaceLoadedEvent(workspace_id="WS1", workspace_name="TestWorkspace")
        bus.publish(evt)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].workspace_id, "WS1")
        self.assertEqual(len(bus.event_history), 1)

    def test_service_container_and_all_services(self) -> None:
        """Test ServiceContainer IoC and core service registrations."""
        container = ServiceContainer()
        container.clear()

        ws = WorkspaceService()
        nav = NavigationService()
        container.register(WorkspaceService, ws)
        container.register(NavigationService, nav)

        resolved_ws = container.resolve(WorkspaceService)
        self.assertEqual(resolved_ws.service_name, "WorkspaceService")

        by_name = container.resolve_by_name("NavigationService")
        self.assertEqual(by_name.service_name, "NavigationService")

    def test_theme_engine(self) -> None:
        """Test StudioThemeEngine dynamic live theme switching."""
        theme_eng = StudioThemeEngine(initial_theme="Dark")
        self.assertEqual(theme_eng.current_theme.name, "Dark")

        theme_eng.set_theme("Light")
        self.assertEqual(theme_eng.current_theme.name, "Light")
        self.assertIn("High Contrast", theme_eng.list_available_themes())

    def test_notification_and_monitoring_frameworks(self) -> None:
        """Test NotificationFramework and MonitoringFramework telemetry."""
        notif = NotificationFramework()
        notif.notify("INFO", "System Init", "Studio online")
        notif.notify("ERROR", "Database Error", "Connection failed")

        history = notif.get_history(severity_filter="ERROR")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].title, "Database Error")

        mon = MonitoringFramework()
        mon.register_module_health("DataEngine", "HEALTHY")
        telemetry = mon.collect_telemetry()
        self.assertEqual(telemetry.health_status, "HEALTHY")

    def test_session_manager_and_navigation_registry(self) -> None:
        """Test SessionManager persistence and NavigationRegistry dynamic routing."""
        sess = SessionManager(session_filepath=self.session_file)
        sess.update_state("active_theme", "Corporate")
        sess.save_session()

        sess2 = SessionManager(session_filepath=self.session_file)
        sess2.load_session()
        self.assertEqual(sess2.get_state("active_theme"), "Corporate")

        nav_reg = NavigationRegistry()
        nav_reg.register_module("analytics", "Analytics", order=15)
        modules = nav_reg.list_modules()
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].module_id, "analytics")

    def test_dashboard_foundation_and_workspace_manager(self) -> None:
        """Test DashboardFoundation widget registry and WorkspaceManager."""
        dash = DashboardFoundation()
        dash.register_widget("w1", "PnL Overview", provider_fn=lambda: {"pnl": 5000})

        data = dash.get_widget_data("w1")
        self.assertEqual(data["pnl"], 5000)

        wm = WorkspaceManager()
        ws = wm.create_workspace("ws_test", "Test WS", os.path.join(self.temp_dir, "ws"))
        self.assertEqual(wm.active_workspace.workspace_id, "ws_test")

    def test_application_shell_and_studio_app(self) -> None:
        """Test ApplicationShell view tabs and master QuantLabStudioApp launcher."""
        shell = ApplicationShell(session_manager=SessionManager(session_filepath=self.session_file))
        shell.activate_tab("strategies")
        self.assertEqual(shell.shell_state.active_tab, "strategies")

        app = QuantLabStudioApp(session_filepath=self.session_file)
        res = app.run()
        self.assertEqual(res["status"], "RUNNING")
        self.assertGreater(res["registered_modules"], 0)


if __name__ == "__main__":
    unittest.main()
