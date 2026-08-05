"""
QuantLab Work Perspective Manager.

Provides work perspective configurations (Research, Machine Learning, Optimization,
Portfolio, Backtesting, Visualization, Development) and custom perspective creation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("PerspectiveManager")


@dataclass
class StudioPerspective:
    """Dataclass holding work perspective panel layout preset."""

    name: str
    description: str
    active_panels: List[str]
    default_active_panel: str


class PerspectiveManager:
    """Institutional Perspective Manager Engine."""

    PREDEFINED_PERSPECTIVES: Dict[str, StudioPerspective] = {
        "Research": StudioPerspective(
            name="Research",
            description="Scientific research, hypotheses, and exploratory analysis",
            active_panels=["explorer", "research_lab", "notebook"],
            default_active_panel="research_lab",
        ),
        "Machine Learning": StudioPerspective(
            name="Machine Learning",
            description="Model training, feature engineering, and evaluation",
            active_panels=["explorer", "ml_lab", "feature_store", "model_registry"],
            default_active_panel="ml_lab",
        ),
        "Optimization": StudioPerspective(
            name="Optimization",
            description="Hyperparameter tuning, Optuna, and Pareto frontiers",
            active_panels=["explorer", "opt_lab", "pareto_view"],
            default_active_panel="opt_lab",
        ),
        "Portfolio": StudioPerspective(
            name="Portfolio",
            description="Asset allocation, Risk Parity, and Monte Carlo simulation",
            active_panels=["explorer", "portfolio_builder", "risk_view"],
            default_active_panel="portfolio_builder",
        ),
        "Backtesting": StudioPerspective(
            name="Backtesting",
            description="Backtest simulation, execution logs, and trade analytics",
            active_panels=["explorer", "backtest_view", "trade_log"],
            default_active_panel="backtest_view",
        ),
        "Visualization": StudioPerspective(
            name="Visualization",
            description="Interactive charting, heatmaps, and financial analytics",
            active_panels=["chart_view", "equity_view", "heatmap_view"],
            default_active_panel="chart_view",
        ),
        "Development": StudioPerspective(
            name="Development",
            description="Full-stack code development and terminal console",
            active_panels=["explorer", "editor", "terminal"],
            default_active_panel="editor",
        ),
    }

    def __init__(self, initial_perspective: str = "Research") -> None:
        self._perspectives: Dict[str, StudioPerspective] = dict(self.PREDEFINED_PERSPECTIVES)
        self.active_perspective_name: str = initial_perspective

    def set_perspective(self, perspective_name: str) -> Optional[StudioPerspective]:
        """Switch active work perspective."""
        if perspective_name in self._perspectives:
            self.active_perspective_name = perspective_name
            p = self._perspectives[perspective_name]
            logger.info(f"Switched work perspective to '{perspective_name}'")
            return p
        return None

    def register_custom_perspective(self, name: str, description: str, active_panels: List[str]) -> StudioPerspective:
        """Register custom user-defined work perspective."""
        p = StudioPerspective(
            name=name,
            description=description,
            active_panels=active_panels,
            default_active_panel=active_panels[0] if active_panels else "dashboard",
        )
        self._perspectives[name] = p
        return p

    def list_perspectives(self) -> List[StudioPerspective]:
        """List registered work perspectives."""
        return list(self._perspectives.values())
