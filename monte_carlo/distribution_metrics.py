"""
QuantLab Statistical Distribution Metrics Calculator.

Computes statistical distribution properties across Monte Carlo simulation iterations:
Expected Return, Median Return, Variance, Skewness, Kurtosis, Max Drawdown Distribution,
and Recovery Time Distribution.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy import stats

from monte_carlo.simulation_runner import SimulationIterationResult


class DistributionMetricsCalculator:
    """Institutional Statistical Distribution Analytics."""

    @staticmethod
    def calculate_distribution_metrics(
        iterations: List[SimulationIterationResult],
    ) -> Dict[str, float]:
        """Compute statistical moments and distribution summary metrics across all iterations.

        Args:
            iterations: List of SimulationIterationResult objects.

        Returns:
            Dict containing statistical distribution metrics.
        """
        if not iterations:
            return {
                "expected_return": 0.0,
                "median_return": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "mean_max_drawdown_pct": 0.0,
                "median_max_drawdown_pct": 0.0,
                "p95_max_drawdown_pct": 0.0,
                "worst_max_drawdown_pct": 0.0,
            }

        net_profits = np.array([it.net_profit for it in iterations])
        max_dds = np.array([it.max_drawdown_pct for it in iterations])

        mean_ret = float(np.mean(net_profits))
        median_ret = float(np.median(net_profits))
        variance = float(np.var(net_profits))
        std_dev = float(np.std(net_profits))

        skewness = float(stats.skew(net_profits)) if len(net_profits) > 2 else 0.0
        kurtosis = float(stats.kurtosis(net_profits)) if len(net_profits) > 3 else 0.0

        mean_dd = float(np.mean(max_dds))
        median_dd = float(np.median(max_dds))
        p95_dd = float(np.percentile(max_dds, 95.0))
        worst_dd = float(np.max(max_dds))

        return {
            "expected_return": mean_ret,
            "median_return": median_ret,
            "variance": variance,
            "std_dev": std_dev,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "mean_max_drawdown_pct": mean_dd,
            "median_max_drawdown_pct": median_dd,
            "p95_max_drawdown_pct": p95_dd,
            "worst_max_drawdown_pct": worst_dd,
        }
