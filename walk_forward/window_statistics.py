"""
QuantLab Window Statistics Calculator.

Computes statistical summaries for each individual Walk Forward window step and aggregates In-Sample vs Out-of-Sample performance comparison matrices.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from walk_forward.validation_runner import ValidationStepResult


class WindowStatisticsCalculator:
    """Institutional Statistics Calculator for Walk Forward Windows."""

    @staticmethod
    def compute_summary_table(step_results: List[ValidationStepResult]) -> pd.DataFrame:
        """Generate DataFrame summarizing each window's IS vs OOS parameters and key performance metrics.

        Args:
            step_results: List of ValidationStepResult objects.

        Returns:
            pd.DataFrame table indexed by window_index.
        """
        if not step_results:
            return pd.DataFrame()

        rows = []
        for res in step_results:
            is_m = res.is_metrics
            oos_m = res.oos_metrics
            row = {
                "window": res.window_index,
                "train_bars": res.train_split.train_bars,
                "val_bars": res.train_split.val_bars,
                "best_params": str(res.best_params),
                "is_net_profit": is_m.get("net_profit", 0.0),
                "is_sharpe": is_m.get("sharpe_ratio", 0.0),
                "is_win_rate": is_m.get("win_rate", 0.0),
                "is_max_dd_pct": is_m.get("max_drawdown_pct", 0.0),
                "oos_net_profit": oos_m.get("net_profit", 0.0),
                "oos_sharpe": oos_m.get("sharpe_ratio", 0.0),
                "oos_win_rate": oos_m.get("win_rate", 0.0),
                "oos_max_dd_pct": oos_m.get("max_drawdown_pct", 0.0),
                "oos_trades": oos_m.get("total_trades", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("window", inplace=True)
        return df

    @staticmethod
    def compute_aggregate_statistics(step_results: List[ValidationStepResult]) -> Dict[str, Any]:
        """Compute aggregate summary stats across all Walk Forward window steps.

        Returns:
            Dict containing mean, median, std dev, min, max for IS vs OOS metrics.
        """
        if not step_results:
            return {}

        df = WindowStatisticsCalculator.compute_summary_table(step_results)

        return {
            "window_count": len(step_results),
            "mean_is_sharpe": float(df["is_sharpe"].mean()),
            "mean_oos_sharpe": float(df["oos_sharpe"].mean()),
            "std_oos_sharpe": float(df["oos_sharpe"].std()) if len(df) > 1 else 0.0,
            "mean_is_net_profit": float(df["is_net_profit"].mean()),
            "mean_oos_net_profit": float(df["oos_net_profit"].mean()),
            "total_oos_net_profit": float(df["oos_net_profit"].sum()),
            "mean_oos_win_rate": float(df["oos_win_rate"].mean()),
            "mean_oos_max_dd_pct": float(df["oos_max_dd_pct"].mean()),
            "max_oos_max_dd_pct": float(df["oos_max_dd_pct"].max()),
            "total_oos_trades": int(df["oos_trades"].sum()),
            "positive_oos_windows": int((df["oos_net_profit"] > 0).sum()),
            "positive_oos_window_pct": float((df["oos_net_profit"] > 0).mean() * 100.0),
        }
