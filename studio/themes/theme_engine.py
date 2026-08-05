"""
QuantLab Studio Theme Engine.

Manages institutional color palettes and UI styling tokens for Dark, Light, High Contrast,
Corporate, and custom themes with dynamic live switching.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("ThemeEngine")


@dataclass
class StudioThemePalette:
    """Dataclass holding theme color tokens."""

    name: str
    bg_primary: str
    bg_secondary: str
    text_primary: str
    accent: str
    border: str
    status_ok: str
    status_warn: str
    status_error: str


class StudioThemeEngine:
    """Institutional Studio Theme Engine."""

    THEMES: Dict[str, StudioThemePalette] = {
        "Dark": StudioThemePalette(
            name="Dark",
            bg_primary="#121212",
            bg_secondary="#1e1e1e",
            text_primary="#ffffff",
            accent="#007acc",
            border="#333333",
            status_ok="#00e676",
            status_warn="#ff9100",
            status_error="#ff1744",
        ),
        "Light": StudioThemePalette(
            name="Light",
            bg_primary="#f5f5f5",
            bg_secondary="#ffffff",
            text_primary="#111111",
            accent="#0066cc",
            border="#cccccc",
            status_ok="#2e7d32",
            status_warn="#f57c00",
            status_error="#d32f2f",
        ),
        "High Contrast": StudioThemePalette(
            name="High Contrast",
            bg_primary="#000000",
            bg_secondary="#000000",
            text_primary="#ffff00",
            accent="#00ffff",
            border="#ffffff",
            status_ok="#00ff00",
            status_warn="#ffaa00",
            status_error="#ff0000",
        ),
        "Corporate": StudioThemePalette(
            name="Corporate",
            bg_primary="#1c2833",
            bg_secondary="#273746",
            text_primary="#eaeded",
            accent="#5d6d7e",
            border="#34495e",
            status_ok="#27ae60",
            status_warn="#f39c12",
            status_error="#c0392b",
        ),
    }

    def __init__(self, initial_theme: str = "Dark") -> None:
        self._current_theme_name: str = initial_theme

    @property
    def current_theme(self) -> StudioThemePalette:
        """Get current theme palette."""
        return self.THEMES.get(self._current_theme_name, self.THEMES["Dark"])

    def set_theme(self, theme_name: str) -> StudioThemePalette:
        """Switch active theme dynamically without application restart.

        Args:
            theme_name: Theme name ('Dark', 'Light', 'High Contrast', 'Corporate').

        Returns:
            New active StudioThemePalette.
        """
        if theme_name in self.THEMES:
            old = self._current_theme_name
            self._current_theme_name = theme_name
            logger.log_theme_change(old, theme_name)
        return self.current_theme

    def list_available_themes(self) -> List[str]:
        """List registered theme names."""
        return list(self.THEMES.keys())
