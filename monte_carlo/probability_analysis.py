"""
QuantLab Probabilistic Risk Analysis.

Calculates Probability of Profit (PoP), Probability of Ruin (PoR),
Probability of Drawdown exceeding target thresholds, Probability of Recovery, and Loss ECDFs.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from monte_carlo.simulation_runner import SimulationIterationResult


class ProbabilityAnalyzer:
    """Institutional Probabilistic Risk Analyzer."""

    @staticmethod
    def calculate_all_probabilities(
        iterations: List[SimulationIterationResult],
        drawdown_thresholds: List[float] = [10.0, 20.0, 30.0, 50.0],
    ) -> Dict[str, Any]:
        """Compute probabilistic risk metrics across all Monte Carlo iterations.

        Args:
            iterations: List of SimulationIterationResult objects.
            drawdown_thresholds: List of drawdown percentages to evaluate.

        Returns:
            Dict containing probability of profit, probability of ruin, and drawdown probabilities.
        """
        if not iterations:
            return {
                "probability_of_profit_pct": 0.0,
                "probability_of_ruin_pct": 0.0,
                "drawdown_probabilities_pct": {},
            }

        total = len(iterations)

        # 1. Probability of Profit (PoP)
        profitable_count = sum(1 for it in iterations if it.net_profit > 0)
        pop_pct = (profitable_count / total) * 100.0

        # 2. Probability of Ruin (PoR)
        ruin_count = sum(1 for it in iterations if it.ruin_occurred)
        por_pct = (ruin_count / total) * 100.0

        # 3. Probability of Drawdown > X%
        dd_probs = {}
        for dd_thresh in drawdown_thresholds:
            breach_count = sum(1 for it in iterations if it.max_drawdown_pct >= dd_thresh)
            dd_probs[f"drawdown_ge_{int(dd_thresh)}pct"] = (breach_count / total) * 100.0

        return {
            "probability_of_profit_pct": pop_pct,
            "probability_of_ruin_pct": por_pct,
            "drawdown_probabilities_pct": dd_probs,
        }

    @staticmethod
    def calculate_ecdf(series: pd.Series) -> pd.DataFrame:
        """Calculate Empirical Cumulative Distribution Function (ECDF).

        Returns:
            pd.DataFrame containing sorted 'value' and 'cumulative_probability'.
        """
        if series.empty:
            return pd.DataFrame(columns=["value", "cumulative_probability"])

        sorted_vals = np.sort(series.values)
        n = len(sorted_vals)
        cum_prob = np.arange(1, n + 1) / n

        return pd.DataFrame({"value": sorted_vals, "cumulative_probability": cum_prob})
