"""
QuantLab Capital Allocation Manager Engine.

Registers capital allocation models, validates sum of allocation weights, and executes
asset allocation strategies.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from portfolio.allocation_models import (
    BaseAllocationModel,
    BlackLittermanModel,
    EqualRiskContributionModel,
    EqualWeightModel,
    HRPAllocationModel,
    KellyAllocationModel,
    MaximumDiversificationModel,
    MeanVarianceModel,
    MinimumVarianceModel,
    RiskParityModel,
)


class AllocationEngine:
    """Institutional Capital Allocation Engine."""

    def __init__(self) -> None:
        """Initialize AllocationEngine with default models registry."""
        self._models: Dict[str, BaseAllocationModel] = {
            "equal_weight": EqualWeightModel(),
            "risk_parity": RiskParityModel(),
            "minimum_variance": MinimumVarianceModel(),
            "maximum_diversification": MaximumDiversificationModel(),
            "kelly": KellyAllocationModel(),
            "hrp": HRPAllocationModel(),
            "black_litterman": BlackLittermanModel(),
            "mean_variance": MeanVarianceModel(),
            "erc": EqualRiskContributionModel(),
        }

    def register_model(self, model_id: str, model: BaseAllocationModel) -> None:
        """Register custom allocation model instance."""
        self._models[model_id.lower()] = model

    def get_model(self, model_id: str) -> BaseAllocationModel:
        """Fetch allocation model by ID."""
        model_id_clean = model_id.lower()
        if model_id_clean not in self._models:
            return self._models["equal_weight"]
        return self._models[model_id_clean]

    def compute_allocation(
        self,
        asset_symbols: List[str],
        model_id: str = "equal_weight",
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute normalized allocation weights dictionary for asset universe.

        Returns:
            Dictionary mapping asset symbol to allocation weight (summing to 1.0).
        """
        model = self.get_model(model_id)
        weights = model.allocate(
            asset_symbols=asset_symbols,
            returns_df=returns_df,
            cov_matrix=cov_matrix,
            expected_returns=expected_returns,
            views_dict=views_dict,
        )

        # Normalize and validate weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            return {k: float(v / total) for k, v in weights.items()}
        n = len(asset_symbols)
        return {sym: 1.0 / n for sym in asset_symbols} if n > 0 else {}
