"""
QuantLab Portfolio Constraints System.

Enforces min/max asset weights, sector exposure limits, maximum leverage ceilings,
liquidity minimums, and rebalancing turnover caps.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PortfolioConstraints:
    """Institutional Portfolio Operational Constraints Specification."""

    min_asset_weight: float = 0.0
    max_asset_weight: float = 1.0
    max_sector_weight: float = 0.40
    max_leverage: float = 2.0
    min_liquidity_ratio: float = 0.10
    max_rebalance_turnover: float = 0.50
    sector_limits: Dict[str, float] = field(default_factory=dict)

    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """Check if asset weights satisfy min/max constraints.

        Args:
            weights: Weights dictionary.

        Returns:
            Boolean indicating validity.
        """
        for w in weights.values():
            if w < self.min_asset_weight - 1e-6 or w > self.max_asset_weight + 1e-6:
                return False
        return True

    def apply_bounds(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clip weights within min_asset_weight and max_asset_weight bounds and renormalize.

        Args:
            weights: Input weights dictionary.

        Returns:
            Clipped and normalized weights dictionary.
        """
        clipped = {
            k: float(min(self.max_asset_weight, max(self.min_asset_weight, v))) for k, v in weights.items()
        }
        total = sum(clipped.values())
        if total > 0:
            return {k: v / total for k, v in clipped.items()}
        n = len(weights)
        return {k: 1.0 / n for k in weights.keys()} if n > 0 else {}
