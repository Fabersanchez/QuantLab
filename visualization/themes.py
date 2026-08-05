"""
QuantLab Institutional Theme Manager.

Provides 5 built-in color themes for scientific financial charts:
1. Dark (Sleek dark mode)
2. Light (Clean light background)
3. Institutional (Executive slate/navy)
4. Trading Desk (High-contrast trading terminal)
5. Research (Academic publication style)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt


@dataclass
class Theme:
    """Dataclass defining chart aesthetic styling properties."""

    name: str
    background_color: str
    paper_color: str
    text_color: str
    grid_color: str
    grid_alpha: float
    bull_color: str
    bear_color: str
    primary_color: str
    secondary_color: str
    accent_color: str
    font_family: str = "sans-serif"
    dpi: int = 150

    def apply_to_ax(self, ax: plt.Axes) -> None:
        """Apply theme styling options to a Matplotlib Axes object."""
        ax.set_facecolor(self.background_color)
        ax.tick_params(colors=self.text_color, labelsize=9)
        ax.xaxis.label.set_color(self.text_color)
        ax.yaxis.label.set_color(self.text_color)
        ax.title.set_color(self.text_color)

        # Spines styling
        for spine in ax.spines.values():
            spine.set_color(self.grid_color)
            spine.set_alpha(0.5)

        ax.grid(True, color=self.grid_color, alpha=self.grid_alpha, linestyle=":")


class ThemeManager:
    """Institutional Theme Manager for QuantLab Graphics."""

    THEMES: Dict[str, Theme] = {
        "dark": Theme(
            name="Dark",
            background_color="#131722",
            paper_color="#1e222d",
            text_color="#d1d4dc",
            grid_color="#2a2e39",
            grid_alpha=0.6,
            bull_color="#089981",
            bear_color="#f23645",
            primary_color="#2962ff",
            secondary_color="#ff6d00",
            accent_color="#00bcd4",
        ),
        "light": Theme(
            name="Light",
            background_color="#ffffff",
            paper_color="#f8f9fa",
            text_color="#191919",
            grid_color="#e0e0e0",
            grid_alpha=0.7,
            bull_color="#26a69a",
            bear_color="#ef5350",
            primary_color="#1e88e5",
            secondary_color="#fb8c00",
            accent_color="#8e24aa",
        ),
        "institutional": Theme(
            name="Institutional",
            background_color="#0f172a",
            paper_color="#1e293b",
            text_color="#f8fafc",
            grid_color="#334155",
            grid_alpha=0.5,
            bull_color="#10b981",
            bear_color="#f43f5e",
            primary_color="#38bdf8",
            secondary_color="#f59e0b",
            accent_color="#a855f7",
        ),
        "trading_desk": Theme(
            name="Trading Desk",
            background_color="#000000",
            paper_color="#0a0a0a",
            text_color="#00ff66",
            grid_color="#1a1a1a",
            grid_alpha=0.8,
            bull_color="#00ff66",
            bear_color="#ff0055",
            primary_color="#00e5ff",
            secondary_color="#ffea00",
            accent_color="#ff00ff",
        ),
        "research": Theme(
            name="Research",
            background_color="#f8fafc",
            paper_color="#ffffff",
            text_color="#0f172a",
            grid_color="#cbd5e1",
            grid_alpha=0.6,
            bull_color="#059669",
            bear_color="#dc2626",
            primary_color="#0284c7",
            secondary_color="#d97706",
            accent_color="#7c3aed",
        ),
    }

    @classmethod
    def get_theme(cls, theme_name: str = "dark") -> Theme:
        """Get Theme object by name identifier.

        Args:
            theme_name: One of 'dark', 'light', 'institutional', 'trading_desk', 'research'.

        Returns:
            Theme instance.
        """
        key = theme_name.lower().replace(" ", "_")
        return cls.THEMES.get(key, cls.THEMES["dark"])

    @classmethod
    def apply(cls, fig: plt.Figure, theme_name: str = "dark") -> Theme:
        """Apply theme background colors and parameters to full Figure object.

        Args:
            fig: Matplotlib Figure instance.
            theme_name: Name of theme.

        Returns:
            Applied Theme instance.
        """
        theme = cls.get_theme(theme_name)
        fig.set_facecolor(theme.paper_color)
        fig.set_dpi(theme.dpi)

        for ax in fig.get_axes():
            theme.apply_to_ax(ax)

        return theme

    @classmethod
    def list_themes(cls) -> List[str]:
        """List all available theme names."""
        return [t.name for t in cls.THEMES.values()]
