"""
QuantLab Institutional Robustness Metrics Calculator.

Computes Walk Forward Efficiency (WFE), Stability Score, Parameter Stability,
Out-of-Sample Ratio, Robustness Index, and Overfitting Score.
"""

import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from walk_forward.validation_runner import ValidationStepResult


class RobustnessMetricsCalculator:
    """Quantitative Strategy Robustness & Overfitting Analyzer."""

    @staticmethod
    def calculate_all(
        step_results: List[ValidationStepResult], total_dataset_bars: int
    ) -> Dict[str, float]:
        """Calculate full suite of Walk Forward robustness metrics.

        Args:
            step_results: List of ValidationStepResult objects from Walk Forward run.
            total_dataset_bars: Total number of bars in original market dataset.

        Returns:
            Dict containing calculated robustness metrics.
        """
        if not step_results:
            return {
                "walk_forward_efficiency_pct": 0.0,
                "stability_score_pct": 0.0,
                "parameter_stability_score": 0.0,
                "out_of_sample_ratio_pct": 0.0,
                "robustness_index": 0.0,
                "overfitting_score_pct": 100.0,
            }

        # 1. Walk Forward Efficiency (WFE %)
        wfe_pct = RobustnessMetricsCalculator.calculate_wfe(step_results)

        # 2. Stability Score (% of positive OOS windows)
        positive_windows = sum(1 for s in step_results if s.oos_metrics.get("net_profit", 0.0) > 0)
        stability_score_pct = (positive_windows / len(step_results)) * 100.0

        # 3. Parameter Stability Score
        param_stability = RobustnessMetricsCalculator.calculate_parameter_stability(step_results)

        # 4. Out-of-Sample Ratio (% of total bars in OOS validation)
        val_bars_total = sum(s.train_split.val_bars for s in step_results)
        oos_ratio_pct = (val_bars_total / total_dataset_bars * 100.0) if total_dataset_bars > 0 else 0.0

        # 5. Overfitting Score (degree of degradation from IS to OOS)
        overfitting_score_pct = max(0.0, min(100.0, 100.0 - wfe_pct))

        # 6. Composite Robustness Index (0 to 100 scale)
        robustness_index = (
            (max(0.0, min(100.0, wfe_pct)) * 0.4)
            + (stability_score_pct * 0.4)
            + (param_stability * 0.2)
        )

        return {
            "walk_forward_efficiency_pct": wfe_pct,
            "stability_score_pct": stability_score_pct,
            "parameter_stability_score": param_stability,
            "out_of_sample_ratio_pct": oos_ratio_pct,
            "robustness_index": robustness_index,
            "overfitting_score_pct": overfitting_score_pct,
        }

    @staticmethod
    def calculate_wfe(step_results: List[ValidationStepResult]) -> float:
        """Calculate Walk Forward Efficiency (WFE %) as ratio of OOS Sharpe to IS Sharpe."""
        if not step_results:
            return 0.0

        mean_is_sharpe = float(np.mean([s.is_metrics.get("sharpe_ratio", 0.0) for s in step_results]))
        mean_oos_sharpe = float(np.mean([s.oos_metrics.get("sharpe_ratio", 0.0) for s in step_results]))

        if mean_is_sharpe <= 0:
            return 100.0 if mean_oos_sharpe > 0 else 0.0

        wfe = (mean_oos_sharpe / mean_is_sharpe) * 100.0
        return max(-100.0, min(300.0, wfe))

    @staticmethod
    def calculate_parameter_stability(step_results: List[ValidationStepResult]) -> float:
        """Calculate Parameter Stability Score (0 to 100 scale).

        Measures numerical stability / low variance of selected hyperparameters across windows.
        """
        if not step_results or len(step_results) < 2:
            return 100.0

        # Extract numeric parameters across steps
        param_keys = list(step_results[0].best_params.keys())
        numeric_keys = [
            k for k in param_keys if isinstance(step_results[0].best_params[k], (int, float))
        ]

        if not numeric_keys:
            return 100.0

        cv_scores = []
        for k in numeric_keys:
            vals = [float(s.best_params[k]) for s in step_results if k in s.best_params]
            if len(vals) > 1 and np.mean(vals) != 0:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                cv = abs(std_val / mean_val)
                cv_scores.append(cv)

        if not cv_scores:
            return 100.0

        avg_cv = float(np.mean(cv_scores))
        stability = max(0.0, 100.0 * (1.0 - min(1.0, avg_cv)))
        return stability
