"""
QuantLab Studio Dashboard Package.
"""

from studio.dashboard.dashboard_engine import DashboardEngine, DashboardWidgetInstance
from studio.dashboard.dashboard_foundation import DashboardFoundation, DashboardWidgetCard

__all__ = [
    "DashboardFoundation",
    "DashboardWidgetCard",
    "DashboardEngine",
    "DashboardWidgetInstance",
]
