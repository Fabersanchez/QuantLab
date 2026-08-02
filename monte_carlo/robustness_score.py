"""
QuantLab Institutional Robustness Score Engine.

Calculates composite Institutional Robustness Score (0 - 100 scale) integrating
Monte Carlo ruin probability, drawdown distribution, stability, sensitivity elasticity, and Walk Forward efficiency.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from monte_carlo.simulation_runner import SimulationIterationResult


class InstitutionalRobustnessScore:
    """Institutional Strategy Robustness Scoring Model."""

    @staticmethod
    def calculate_score(
        monte_carlo_results: List[SimulationIterationResult],
        walk_forward_metrics: Optional[Dict[str, float]] = None,
        sensitivity_elasticity: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate composite institutional robustness score and sub-scores.

        Args:
            monte_carlo_results: List of SimulationIterationResult objects.
            walk_forward_metrics: Optional dictionary from Walk Forward robustness calculator.
            sensitivity_elasticity: Optional sensitivity elasticity score (0 to 100).

        Returns:
            Dict containing composite robustness score (0 to 100) and sub-component breakdown.
        """
        if not monte_carlo_results:
            return {
                "institutional_robustness_score": 0.0,
                "profitability_stability_score": 0.0,
                "drawdown_resilience_score": 0.0,
                "ruin_safety_score": 0.0,
                "mc_consistency_score": 0.0,
                "sensitivity_score": 0.0,
                "walk_forward_score": 0.0,
            }

        n = len(monte_carlo_results)

        # 1. Profitability Stability Score (% profitable iterations)
        profitable_count = sum(1 for it in monte_carlo_results if it.net_profit > 0)
        prof_score = (profitable_count / n) * 100.0

        # 2. Drawdown Resilience Score (penalizes mean & worst drawdown)
        mean_dd = float(np.mean([it.max_drawdown_pct for it in monte_carlo_results]))
        dd_score = max(0.0, 100.0 - (mean_dd * 1.5))

        # 3. Ruin Safety Score (% without ruin)
        ruin_count = sum(1 for it in monte_carlo_results if it.ruin_occurred)
        ruin_safety_score = max(0.0, 100.0 - (ruin_count / n * 100.0 * 2.0))

        # 4. Monte Carlo Consistency Score (low variance of returns)
        returns = [it.net_profit for it in monte_carlo_results]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        cv = (std_ret / abs(mean_ret)) if mean_ret != 0 else 2.0
        consistency_score = max(0.0, 100.0 * (1.0 - min(1.0, cv / 2.0)))

        # 5. Sensitivity Score
        sens_score = 100.0 - min(100.0, sensitivity_elasticity) if sensitivity_elasticity is not None else 80.0

        # 6. Walk Forward Score
        wf_score = 80.0
        if walk_forward_metrics:
            wf_score = float(walk_forward_metrics.get("robustness_index", 80.0))

        # Weighted Composite Institutional Robustness Score
        composite_score = (
            (prof_score * 0.20)
            + (dd_score * 0.20)
            + (ruin_safety_score * 0.20)
            + (consistency_score * 0.15)
            + (sens_score * 0.15)
            + (wf_score * 0.10)
        )

        composite_score = max(0.0, min(100.0, composite_score))

        return {
            "institutional_robustness_score": composite_score,
            "profitability_stability_score": prof_score,
            "drawdown_resilience_score": dd_score,
            "ruin_safety_score": ruin_safety_score,
            "mc_consistency_score": consistency_score,
            "sensitivity_score": sens_score,
            "walk_forward_score": wf_score,
        }
