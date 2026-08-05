"""
QuantLab Studio Application Shell Engine.

Acts as the primary UI Shell container orchestrating top bar, sidebar, dynamic workspace tabs,
docking panels, bottom status bar, panel layout manager, and session restoration state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from studio.events.event_bus import StudioEventBus
from studio.events.studio_events import ViewChangedEvent
from studio.logging.studio_logger import get_studio_logger
from studio.navigation.navigation_registry import NavigationRegistry
from studio.settings.session_manager import SessionManager
from studio.themes.theme_engine import StudioThemeEngine

logger = get_studio_logger("ApplicationShell")


@dataclass
class ShellState:
    """Dataclass holding Application Shell UI layout state."""

    top_bar_title: str = "QuantLab Studio Enterprise"
    active_sidebar_item: str = "dashboard"
    open_tabs: List[str] = field(default_factory=lambda: ["dashboard"])
    active_tab: str = "dashboard"
    bottom_status_msg: str = "QuantLab Studio Ready"
    is_sidebar_expanded: bool = True


class ApplicationShell:
    """Institutional Application Shell Container Engine."""

    def __init__(
        self,
        event_bus: Optional[StudioEventBus] = None,
        navigation_registry: Optional[NavigationRegistry] = None,
        session_manager: Optional[SessionManager] = None,
        theme_engine: Optional[StudioThemeEngine] = None,
    ) -> None:
        self.event_bus = event_bus or StudioEventBus()
        self.navigation_registry = navigation_registry or NavigationRegistry()
        self.session_manager = session_manager or SessionManager()
        self.theme_engine = theme_engine or StudioThemeEngine()

        self.shell_state = ShellState()
        self._restore_session_state()

    def _restore_session_state(self) -> None:
        """Restore shell state from session manager."""
        state = self.session_manager.load_session()
        if "active_theme" in state:
            self.theme_engine.set_theme(state["active_theme"])
        if "open_panels" in state:
            self.shell_state.open_tabs = state["open_panels"]
        if "active_view" in state:
            self.shell_state.active_tab = state["active_view"]

    def activate_tab(self, tab_id: str) -> None:
        """Activate target tab in shell dynamic area."""
        old = self.shell_state.active_tab
        if tab_id not in self.shell_state.open_tabs:
            self.shell_state.open_tabs.append(tab_id)
        self.shell_state.active_tab = tab_id
        self.shell_state.active_sidebar_item = tab_id

        self.session_manager.update_state("active_view", tab_id)
        self.session_manager.update_state("open_panels", self.shell_state.open_tabs)
        self.session_manager.save_session()

        self.event_bus.publish(ViewChangedEvent(old_view_id=old, new_view_id=tab_id))
        logger.info(f"Shell view changed: '{old}' -> '{tab_id}'")

    def close_tab(self, tab_id: str) -> None:
        """Close open tab from shell dynamic area."""
        if tab_id in self.shell_state.open_tabs:
            self.shell_state.open_tabs.remove(tab_id)
            if self.shell_state.active_tab == tab_id:
                self.shell_state.active_tab = (
                    self.shell_state.open_tabs[-1] if self.shell_state.open_tabs else "dashboard"
                )

        self.session_manager.update_state("open_panels", self.shell_state.open_tabs)
        self.session_manager.save_session()

    def set_status_message(self, message: str) -> None:
        """Update bottom status bar message."""
        self.shell_state.bottom_status_msg = message
