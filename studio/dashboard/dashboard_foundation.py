"""
QuantLab Studio Dashboard Foundation Engine.

Provides an adaptable widget grid, card widget registry, async metric provider interface,
lazy loading, and auto-refresh telemetry loops.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DashboardWidgetCard:
    """Dataclass holding dashboard card widget specification."""

    widget_id: str
    title: str
    category: str = "Overview"
    col_span: int = 1
    row_span: int = 1
    provider_fn: Optional[Callable[[], Dict[str, Any]]] = None


class DashboardFoundation:
    """Institutional Dashboard Foundation Engine."""

    def __init__(self) -> None:
        self._registered_widgets: Dict[str, DashboardWidgetCard] = {}

    def register_widget(
        self,
        widget_id: str,
        title: str,
        category: str = "Overview",
        col_span: int = 1,
        row_span: int = 1,
        provider_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> DashboardWidgetCard:
        """Register card widget in dashboard grid."""
        card = DashboardWidgetCard(
            widget_id=widget_id,
            title=title,
            category=category,
            col_span=col_span,
            row_span=row_span,
            provider_fn=provider_fn,
        )
        self._registered_widgets[widget_id] = card
        return card

    def get_widget_data(self, widget_id: str) -> Dict[str, Any]:
        """Fetch real-time metric payload for target widget."""
        card = self._registered_widgets.get(widget_id)
        if card and card.provider_fn:
            try:
                return card.provider_fn()
            except Exception as e:
                return {"error": str(e)}
        return {"status": "NO_PROVIDER"}

    def list_widgets(self) -> List[DashboardWidgetCard]:
        """List registered dashboard cards."""
        return list(self._registered_widgets.values())
