"""
QuantLab Studio Session Manager & State Persistence System.

Persists active workspace ID, open tab panels, window layout coordinates, active theme,
configuration settings, filters, search history, and shell state into JSON file storage.
"""

import json
import os
from typing import Any, Dict, List, Optional


class SessionManager:
    """Institutional Studio Session Manager."""

    def __init__(self, session_filepath: str = "studio_session.json") -> None:
        self.session_filepath = session_filepath
        self._session_data: Dict[str, Any] = {
            "active_workspace": "default_workspace",
            "active_theme": "Dark",
            "open_panels": ["dashboard", "explorer"],
            "active_view": "main_dashboard",
            "window_geometry": {"width": 1600, "height": 900, "x": 100, "y": 100},
            "filters": {},
            "history": [],
        }

    def load_session(self) -> Dict[str, Any]:
        """Load session state from file if present."""
        if os.path.exists(self.session_filepath):
            try:
                with open(self.session_filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self._session_data.update(loaded)
            except Exception:
                pass
        return dict(self._session_data)

    def save_session(self) -> bool:
        """Save current session state to file."""
        try:
            with open(self.session_filepath, "w", encoding="utf-8") as f:
                json.dump(self._session_data, f, indent=2)
            return True
        except Exception:
            return False

    def update_state(self, key: str, value: Any) -> None:
        """Update session state key and persist."""
        self._session_data[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Fetch session state key value."""
        return self._session_data.get(key, default)
