"""
QuantLab Enterprise Layout Engine.

Provides docking management, floating window tracking, tabbed document areas,
split panels, layout customization, and per-workspace layout persistence.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class LayoutPanelConfig:
    """Dataclass holding individual panel layout configuration."""

    panel_id: str
    area: str = "main"  # 'main', 'left_sidebar', 'right_sidebar', 'bottom_dock'
    is_floating: bool = False
    is_visible: bool = True
    split_ratio: float = 0.5


@dataclass
class StudioLayoutProfile:
    """Dataclass holding complete UI layout profile."""

    layout_name: str = "DefaultLayout"
    workspace_id: str = "default_workspace"
    panels: Dict[str, LayoutPanelConfig] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert layout profile to dictionary."""
        return {
            "layout_name": self.layout_name,
            "workspace_id": self.workspace_id,
            "panels": {k: asdict(v) for k, v in self.panels.items()},
        }


class LayoutEngine:
    """Institutional Enterprise Layout Engine."""

    def __init__(self) -> None:
        self._layouts: Dict[str, StudioLayoutProfile] = {}
        self.active_layout_name: str = "DefaultLayout"

        # Initialize default layout profile
        default_profile = StudioLayoutProfile(layout_name="DefaultLayout")
        default_profile.panels["explorer"] = LayoutPanelConfig(panel_id="explorer", area="left_sidebar")
        default_profile.panels["dashboard"] = LayoutPanelConfig(panel_id="dashboard", area="main")
        default_profile.panels["terminal"] = LayoutPanelConfig(panel_id="terminal", area="bottom_dock")
        self._layouts["DefaultLayout"] = default_profile

    def get_active_layout(self) -> StudioLayoutProfile:
        """Fetch active StudioLayoutProfile."""
        return self._layouts.get(self.active_layout_name, self._layouts["DefaultLayout"])

    def set_panel_area(self, panel_id: str, area: str, is_floating: bool = False) -> None:
        """Update panel docking area or floating state in active layout."""
        layout = self.get_active_layout()
        if panel_id in layout.panels:
            layout.panels[panel_id].area = area
            layout.panels[panel_id].is_floating = is_floating
        else:
            layout.panels[panel_id] = LayoutPanelConfig(panel_id=panel_id, area=area, is_floating=is_floating)

    def save_layout_to_file(self, filepath: str) -> bool:
        """Export active layout configuration to JSON file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.get_active_layout().to_dict(), f, indent=2)
            return True
        except Exception:
            return False
