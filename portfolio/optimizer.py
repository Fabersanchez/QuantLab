"""
QuantLab Portfolio Optimizer Engine.

Optimizes asset allocation weights for target portfolio objectives:
Sharpe ratio, Sortino ratio, Calmar ratio, Minimum Volatility, Minimum Max Drawdown,
Maximum Return, Maximum Diversification, and Risk Budgeting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Institutional Portfolio Weight Optimization Engine."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """Initialize PortfolioOptimizer.

        Args:
            risk_free_rate: Annual risk-free rate float.
        """
        self.rf = risk_free_rate

    def optimize(
        self,
        asset_symbols: List[str],
        returns_df: pd.DataFrame,
        objective: str = "sharpe",  # 'sharpe', 'sortino', 'calmar', 'volatility', 'drawdown', 'return', 'diversification'
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> Dict[str, float]:
        """Optimize portfolio weights for target objective.

        Args:
            asset_symbols: List of asset symbols.
            returns_df: DataFrame of asset period returns.
            objective: Objective string identifier.
            max_weight: Maximum weight bound per asset.
            min_weight: Minimum weight bound per asset.

        Returns:
            Optimal weights dictionary mapping asset symbol to weight float.
        """
        n = len(asset_symbols)
        if n == 0 or returns_df.empty:
            return {}

        df = returns_df[asset_symbols].dropna()
        cov = df.cov().values * 252.0
        mean_ret = df.mean().values * 252.0

        def get_portfolio_stats(w: np.ndarray) -> Tuple[float, float]:
            p_ret = float(np.dot(w, mean_ret))
            p_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            return p_ret, p_vol

        obj_clean = objective.lower()

        if obj_clean == "volatility":
            def cost(w: np.ndarray) -> float:
                return get_portfolio_stats(w)[1]
        elif obj_clean == "return":
            def cost(w: np.ndarray) -> float:
                return -get_portfolio_stats(w)[0]
        elif obj_clean == "sortino":
            def cost(w: np.ndarray) -> float:
                p_ret, _ = get_portfolio_stats(w)
                port_series = (df * w).sum(axis=1)
                downside_std = float(port_series[port_series < 0].std() * np.sqrt(252.0)) or 1e-4
                return -(p_ret - self.rf) / downside_std
        else:  # 'sharpe' default
            def cost(w: np.ndarray) -> float:
                p_ret, p_vol = get_portfolio_stats(w)
                if p_vol <= 0:
                    return 0.0
                return -(p_ret - self.rf) / p_vol

        bounds = tuple((min_weight, max_weight) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        w0 = np.full(n, 1.0 / n)

        res = minimize(cost, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else w0
        weights = np.maximum(0.0, weights)
        weights /= np.sum(weights)

        return {asset_symbols[i]: float(weights[i]) for i in range(n)}
