"""
QuantLab Efficiency & Drift Analyzer.

Computes Walk Forward Efficiency ratios (Return, Sharpe, Profit Factor WFE),
Rolling CAGR across concatenated OOS windows, Average OOS Return/Drawdown, and Parameter Drift.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from walk_forward.validation_runner import ValidationStepResult


class EfficiencyAnalyzer:
    """Institutional Walk Forward Efficiency Analyzer."""

    @staticmethod
    def analyze_efficiency(
        step_results: List[ValidationStepResult],
        concatenated_oos_equity: pd.DataFrame,
        initial_capital: float = 100000.0,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """Perform complete efficiency breakdown across IS vs OOS validation steps.

        Args:
            step_results: List of ValidationStepResult objects.
            concatenated_oos_equity: DataFrame of stitched OOS equity curve.
            initial_capital: Account starting capital.
            periods_per_year: Bars/periods per trading year.

        Returns:
            Dict containing efficiency metrics and parameter drift.
        """
        if not step_results:
            return {}

        # 1. Individual WFE Metrics
        is_rets = [s.is_metrics.get("total_return_pct", 0.0) for s in step_results]
        oos_rets = [s.oos_metrics.get("total_return_pct", 0.0) for s in step_results]
        return_wfe = (float(np.mean(oos_rets)) / float(np.mean(is_rets)) * 100.0) if float(np.mean(is_rets)) != 0 else 0.0

        is_sharpes = [s.is_metrics.get("sharpe_ratio", 0.0) for s in step_results]
        oos_sharpes = [s.oos_metrics.get("sharpe_ratio", 0.0) for s in step_results]
        sharpe_wfe = (float(np.mean(oos_sharpes)) / float(np.mean(is_sharpes)) * 100.0) if float(np.mean(is_sharpes)) != 0 else 0.0

        is_pfs = [s.is_metrics.get("profit_factor", 0.0) for s in step_results]
        oos_pfs = [s.oos_metrics.get("profit_factor", 0.0) for s in step_results]
        pf_wfe = (float(np.mean(oos_pfs)) / float(np.mean(is_pfs)) * 100.0) if float(np.mean(is_pfs)) != 0 else 0.0

        # 2. Rolling CAGR on concatenated OOS equity curve
        rolling_cagr = 0.0
        if not concatenated_oos_equity.empty and "equity" in concatenated_oos_equity.columns:
            final_eq = float(concatenated_oos_equity["equity"].iloc[-1])
            total_bars = len(concatenated_oos_equity)
            years = total_bars / periods_per_year
            if years > 0 and initial_capital > 0 and final_eq > 0:
                rolling_cagr = (final_eq / initial_capital) ** (1.0 / years) - 1.0

        # 3. Average OOS Return & Drawdown
        avg_oos_return = float(np.mean(oos_rets))
        avg_oos_drawdown = float(np.mean([s.oos_metrics.get("max_drawdown_pct", 0.0) for s in step_results]))

        # 4. Parameter Drift
        param_drift = EfficiencyAnalyzer.calculate_parameter_drift(step_results)

        return {
            "return_wfe_pct": return_wfe,
            "sharpe_wfe_pct": sharpe_wfe,
            "profit_factor_wfe_pct": pf_wfe,
            "rolling_cagr": rolling_cagr,
            "average_oos_return_pct": avg_oos_return,
            "average_oos_drawdown_pct": avg_oos_drawdown,
            "parameter_drift": param_drift,
        }

    @staticmethod
    def calculate_parameter_drift(step_results: List[ValidationStepResult]) -> Dict[str, float]:
        """Compute parameter drift (step-to-step percentage change) across windows.

        Returns:
            Dict mapping parameter_name -> mean_drift_percentage.
        """
        if not step_results or len(step_results) < 2:
            return {}

        param_keys = list(step_results[0].best_params.keys())
        drift_dict: Dict[str, float] = {}

        for k in param_keys:
            vals = [float(s.best_params[k]) for s in step_results if k in s.best_params and isinstance(s.best_params[k], (int, float))]
            if len(vals) > 1:
                drifts = []
                for i in range(1, len(vals)):
                    prev_v = vals[i - 1]
                    curr_v = vals[i]
                    if prev_v != 0:
                        drift = abs(curr_v - prev_v) / abs(prev_v) * 100.0
                    else:
                        drift = abs(curr_v - prev_v) * 100.0
                    drifts.append(drift)

                drift_dict[k] = float(np.mean(drifts)) if drifts else 0.0

        return drift_dict
