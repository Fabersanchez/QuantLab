"""
QuantLab Enterprise Dashboard & Workspace Platform Test Suite.

Validates all Phase 19.2 components:
EnterpriseWorkspace, WorkspaceManager, EnterpriseProject, ProjectManager, ProjectExplorer, ExplorerNode,
DashboardEngine, DashboardWidgetInstance, LayoutEngine, StudioLayoutProfile, BaseWidget, GenericStudioWidget,
WidgetFramework, PerspectiveManager, StudioPerspective, SessionRecoveryEngine, CrashRecoveryCheckpoint,
and StateSynchronizer.
"""

import os
import shutil
import tempfile
import unittest

from studio import (
    CrashRecoveryCheckpoint,
    DashboardEngine,
    EnterpriseProject,
    EnterpriseWorkspace,
    ExplorerNode,
    GenericStudioWidget,
    LayoutEngine,
    PerspectiveManager,
    ProjectExplorer,
    ProjectManager,
    SessionRecoveryEngine,
    StateSynchronizer,
    StudioEventBus,
    WidgetFramework,
    WorkspaceManager,
)


class TestQuantLabWorkspacePlatform(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Enterprise Workspace Platform."""

    def setUp(self) -> None:
        """Set up temporary test directories."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_ws_test_")
        self.proj_dir = os.path.join(self.temp_dir, "projects")
        self.recovery_file = os.path.join(self.temp_dir, "recovery.json")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_workspace_manager_and_integrity(self) -> None:
        """Test WorkspaceManager multi-workspace, cloning, export/import, and SHA-256 integrity."""
        bus = StudioEventBus()
        wm = WorkspaceManager(event_bus=bus)

        ws1 = wm.create_workspace("ws_1", "Alpha Trading Workspace", os.path.join(self.temp_dir, "ws1"))
        self.assertTrue(wm.validate_integrity(ws1.workspace_id))

        ws_clone = wm.clone_workspace("ws_1", "ws_clone", "Cloned Workspace", os.path.join(self.temp_dir, "ws_clone"))
        self.assertIsNotNone(ws_clone)
        self.assertEqual(ws_clone.name, "Cloned Workspace")

        export_path = os.path.join(self.temp_dir, "ws_export.json")
        self.assertTrue(wm.export_workspace("ws_1", export_path))

        imported_ws = wm.import_workspace(export_path)
        self.assertIsNotNone(imported_ws)

    def test_project_manager_and_explorer(self) -> None:
        """Test ProjectManager CRUD operations and ProjectExplorer tree navigation."""
        pm = ProjectManager(root_projects_dir=self.proj_dir)
        proj = pm.create_project("MacroTrendSystem", tags=["forex", "trend"])
        self.assertTrue(os.path.exists(os.path.join(proj.path, proj.strategies_dir)))

        search_results = pm.search_projects("macro")
        self.assertEqual(len(search_results), 1)

        dup = pm.duplicate_project(proj.project_id, "MacroTrendSystem_V2")
        self.assertIsNotNone(dup)

        explorer = ProjectExplorer(root_path=self.proj_dir)
        tree = explorer.build_tree(max_depth=2)
        self.assertIsNotNone(tree)

        explorer.toggle_favorite(proj.path)
        self.assertIn(os.path.abspath(proj.path), explorer.favorites)

    def test_dashboard_engine_and_layouts(self) -> None:
        """Test DashboardEngine dynamic self-registration and LayoutEngine."""
        bus = StudioEventBus()
        dash = DashboardEngine(event_bus=bus)
        w_inst = dash.register_widget("metric_pnl", "PnL Tracker", metric_provider=lambda: {"net_pnl": 12500})

        payload = dash.get_widget_payload("metric_pnl")
        self.assertEqual(payload["net_pnl"], 12500)

        dash.update_widget_geometry("metric_pnl", 50, 50, 500, 400)
        self.assertEqual(w_inst.width, 500)

        layout_eng = LayoutEngine()
        profile = layout_eng.get_active_layout()
        self.assertIn("explorer", profile.panels)

        layout_eng.set_panel_area("terminal", "bottom_dock", is_floating=False)
        self.assertEqual(profile.panels["terminal"].area, "bottom_dock")

    def test_widget_framework_lifecycle(self) -> None:
        """Test WidgetFramework lifecycle methods (Initialize -> Load -> Activate -> Destroy)."""
        wf = WidgetFramework()
        widget = GenericStudioWidget("w_chart", "Candlestick Chart")

        wf.register_widget(widget)
        self.assertTrue(widget.is_loaded)
        self.assertEqual(widget.state, "LOADED")

        wf.activate_widget("w_chart")
        self.assertTrue(widget.is_active)
        self.assertEqual(widget.state, "ACTIVE")

        widget.suspend()
        self.assertEqual(widget.state, "SUSPENDED")

        widget.destroy()
        self.assertFalse(widget.is_active)

    def test_perspective_manager(self) -> None:
        """Test PerspectiveManager work perspective switching and custom perspectives."""
        pm = PerspectiveManager(initial_perspective="Research")
        self.assertEqual(pm.active_perspective_name, "Research")

        p_ml = pm.set_perspective("Machine Learning")
        self.assertIsNotNone(p_ml)
        self.assertEqual(p_ml.name, "Machine Learning")

        custom_p = pm.register_custom_perspective("Execution", "Live trading panel", ["order_entry", "execution_log"])
        self.assertEqual(custom_p.name, "Execution")

    def test_session_recovery_and_state_synchronizer(self) -> None:
        """Test SessionRecoveryEngine checkpoint snapshotting and StateSynchronizer."""
        recovery = SessionRecoveryEngine(recovery_filepath=self.recovery_file)
        cp = recovery.create_checkpoint(active_workspace="ws1", open_files=["strategy1.py"])
        self.assertIsNotNone(cp)

        recovered_cp = recovery.recover_last_session()
        self.assertIsNotNone(recovered_cp)
        self.assertEqual(recovered_cp.active_workspace, "ws1")

        bus = StudioEventBus()
        sync = StateSynchronizer(event_bus=bus)
        sync.set_state("active_mode", "LIVE")
        self.assertEqual(sync.get_state("active_mode"), "LIVE")


if __name__ == "__main__":
    unittest.main()
