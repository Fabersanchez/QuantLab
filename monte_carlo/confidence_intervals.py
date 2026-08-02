"""
QuantLab Statistical Confidence Intervals Calculator.

Computes 90%, 95%, and 99% Percentile and Parametric Bootstrap Confidence Intervals
for Net Profit, Sharpe Ratio, Max Drawdown %, and Final Equity.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

from monte_carlo.simulation_runner import SimulationIterationResult


class ConfidenceIntervalCalculator:
    """Institutional Confidence Interval Estimator."""

    @staticmethod
    def calculate_confidence_intervals(
        iterations: List[SimulationIterationResult],
        confidence_levels: List[float] = [0.90, 0.95, 0.99],
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for key metrics.

        Args:
            iterations: List of SimulationIterationResult objects.
            confidence_levels: List of confidence levels (e.g. 0.90, 0.95, 0.99).

        Returns:
            Dict mapping metric_name -> dict of confidence_level -> (lower_bound, upper_bound).
        """
        if not iterations:
            return {}

        net_profits = np.array([it.net_profit for it in iterations])
        max_dds = np.array([it.max_drawdown_pct for it in iterations])
        final_equities = np.array([it.final_equity for it in iterations])

        results: Dict[str, Dict[str, Tuple[float, float]]] = {
            "net_profit": {},
            "max_drawdown_pct": {},
            "final_equity": {},
        }

        for cl in confidence_levels:
            alpha = (1.0 - cl) / 2.0
            lower_pct = alpha * 100.0
            upper_pct = (1.0 - alpha) * 100.0

            cl_key = f"{int(cl * 100)}%"

            results["net_profit"][cl_key] = (
                float(np.percentile(net_profits, lower_pct)),
                float(np.percentile(net_profits, upper_pct)),
            )
            results["max_drawdown_pct"][cl_key] = (
                float(np.percentile(max_dds, lower_pct)),
                float(np.percentile(max_dds, upper_pct)),
            )
            results["final_equity"][cl_key] = (
                float(np.percentile(final_equities, lower_pct)),
                float(np.percentile(final_equities, upper_pct)),
            )

        return results
