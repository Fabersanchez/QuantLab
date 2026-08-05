"""
QuantLab Studio Central Navigation Registry.

Provides a decoupled, extensible module routing registry that allows dynamic registration
of new modules, views, and navigation endpoints without modifying existing code.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("NavigationRegistry")


@dataclass
class NavigationItem:
    """Dataclass holding module navigation route entry."""

    module_id: str
    title: str
    icon: str = "default_icon"
    category: str = "Core"
    order: int = 100
    is_enabled: bool = True
    view_factory: Optional[Callable[[], Any]] = None


class NavigationRegistry:
    """Institutional Navigation Routing Registry Engine."""

    def __init__(self) -> None:
        self._items: Dict[str, NavigationItem] = {}

    def register_module(
        self,
        module_id: str,
        title: str,
        icon: str = "default_icon",
        category: str = "Core",
        order: int = 100,
        view_factory: Optional[Callable[[], Any]] = None,
    ) -> NavigationItem:
        """Register a new module navigation route dynamically."""
        item = NavigationItem(
            module_id=module_id,
            title=title,
            icon=icon,
            category=category,
            order=order,
            view_factory=view_factory,
        )
        self._items[module_id] = item
        logger.info(f"Registered navigation route for module '{module_id}' ({title})")
        return item

    def get_module(self, module_id: str) -> Optional[NavigationItem]:
        """Fetch registered navigation item by module ID."""
        return self._items.get(module_id)

    def list_modules(self) -> List[NavigationItem]:
        """List all registered navigation items ordered by priority."""
        items = list(self._items.values())
        items.sort(key=lambda x: (x.order, x.title))
        return items
