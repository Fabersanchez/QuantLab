"""
QuantLab Institutional Experiment Comparator.

Compares two or more quantitative research experiments across profitability, risk metrics,
drawdowns, expectancy, latency, and hardware resource consumption. Generates normalized
composite leaderboard rankings and metric breakdown tables.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from research.experiment import Experiment
from research.logger import get_research_logger

logger = get_research_logger("Comparator")


@dataclass
class ComparisonResult:
    """Dataclass holding detailed experiment comparison and benchmark outputs."""

    experiments_count: int
    winner_uuid: str
    winner_name: str
    rankings: List[Dict[str, Any]]
    metrics_comparison_table: pd.DataFrame
    resource_comparison_table: pd.DataFrame
    summary_notes: List[str]


class Comparator:
    """Institutional Comparator engine for multi-experiment benchmarking."""

    def __init__(self, default_weights: Optional[Dict[str, float]] = None) -> None:
        """Initialize Comparator with customizable metric normalization weights.

        Args:
            default_weights: Optional dict of metric weight multipliers.
        """
        self.weights = default_weights or {
            "net_profit": 0.15,
            "total_return": 0.10,
            "cagr": 0.10,
            "profit_factor": 0.15,
            "sharpe_ratio": 0.15,
            "sortino_ratio": 0.10,
            "calmar_ratio": 0.10,
            "recovery_factor": 0.05,
            "expectancy": 0.05,
            "max_drawdown": 0.05,  # Penalty metric
        }

    def _extract_metric(self, exp: Experiment, key: str, default: float = 0.0) -> float:
        """Helper to extract metric value safely from experiment results or metrics dict.

        Args:
            exp: Target experiment instance.
            key: Metric key name.
            default: Default value if missing.

        Returns:
            Float metric value.
        """
        res = exp.results or {}
        if key in res:
            val = res[key]
            return float(val) if isinstance(val, (int, float)) else default
        metrics = res.get("metrics", {})
        if key in metrics:
            val = metrics[key]
            return float(val) if isinstance(val, (int, float)) else default
        return default

    def compare(self, experiments: List[Experiment]) -> ComparisonResult:
        """Compare multiple experiments, compute composite scores, and produce a leaderboard.

        Args:
            experiments: List of two or more Experiment instances to compare.

        Returns:
            ComparisonResult object containing full comparative tables and winner selection.
        """
        if not experiments or len(experiments) < 2:
            raise ValueError("Comparator requires at least 2 experiments to perform benchmark comparison.")

        exp_uuids = [e.uuid for e in experiments]
        logger.info(f"Comparing {len(experiments)} experiments...")

        metrics_data = []
        resource_data = []

        for exp in experiments:
            net_profit = self._extract_metric(exp, "net_profit")
            total_return = self._extract_metric(exp, "total_return")
            cagr = self._extract_metric(exp, "cagr")
            profit_factor = self._extract_metric(exp, "profit_factor")
            sharpe = self._extract_metric(exp, "sharpe_ratio")
            sortino = self._extract_metric(exp, "sortino_ratio")
            calmar = self._extract_metric(exp, "calmar_ratio")
            recovery = self._extract_metric(exp, "recovery_factor")
            expectancy = self._extract_metric(exp, "expectancy")
            max_dd = self._extract_metric(exp, "max_drawdown")

            res_usage = exp.resource_metrics or {}
            ram_mb = float(res_usage.get("ram_peak_mb", 0.0))
            cpu_pct = float(res_usage.get("cpu_usage_pct", 0.0))

            metrics_data.append(
                {
                    "uuid": exp.uuid,
                    "name": exp.name,
                    "version": exp.version,
                    "net_profit": net_profit,
                    "total_return": total_return,
                    "cagr": cagr,
                    "profit_factor": profit_factor,
                    "sharpe_ratio": sharpe,
                    "sortino_ratio": sortino,
                    "calmar_ratio": calmar,
                    "recovery_factor": recovery,
                    "expectancy": expectancy,
                    "max_drawdown": max_dd,
                }
            )

            resource_data.append(
                {
                    "uuid": exp.uuid,
                    "name": exp.name,
                    "execution_time_sec": exp.execution_time,
                    "ram_peak_mb": ram_mb,
                    "cpu_usage_pct": cpu_pct,
                }
            )

        df_metrics = pd.DataFrame(metrics_data)
        df_resources = pd.DataFrame(resource_data)

        # Calculate composite normalized score per experiment
        scores: Dict[str, float] = {e.uuid: 0.0 for e in experiments}

        for metric_name, weight in self.weights.items():
            if metric_name not in df_metrics.columns:
                continue
            vals = df_metrics[metric_name].values
            min_val, max_val = float(vals.min()), float(vals.max())
            val_range = max_val - min_val

            for _, row in df_metrics.iterrows():
                u = row["uuid"]
                v = float(row[metric_name])
                if val_range > 0:
                    norm_score = (v - min_val) / val_range
                else:
                    norm_score = 1.0

                # Max drawdown is a penalty metric
                if metric_name == "max_drawdown":
                    norm_score = 1.0 - norm_score

                scores[u] += norm_score * weight * 100.0

        # Build ranking list
        rankings = []
        for exp in experiments:
            rankings.append(
                {
                    "rank": 0,
                    "uuid": exp.uuid,
                    "name": exp.name,
                    "composite_score": round(scores[exp.uuid], 2),
                    "net_profit": self._extract_metric(exp, "net_profit"),
                    "sharpe_ratio": self._extract_metric(exp, "sharpe_ratio"),
                    "max_drawdown": self._extract_metric(exp, "max_drawdown"),
                    "execution_time": exp.execution_time,
                }
            )

        rankings.sort(key=lambda r: r["composite_score"], reverse=True)
        for idx, r in enumerate(rankings):
            r["rank"] = idx + 1

        winner_uuid = rankings[0]["uuid"]
        winner_name = rankings[0]["name"]

        summary_notes = [
            f"Evaluated {len(experiments)} candidate experiments.",
            f"Winner experiment: '{winner_name}' (UUID={winner_uuid}) with composite score {rankings[0]['composite_score']}/100.",
            f"Top Sharpe Ratio: {df_metrics['sharpe_ratio'].max():.2f}",
            f"Lowest Max Drawdown: {df_metrics['max_drawdown'].min():.2f}%",
        ]

        logger.log_comparison(exp_uuids, winner_uuid)

        return ComparisonResult(
            experiments_count=len(experiments),
            winner_uuid=winner_uuid,
            winner_name=winner_name,
            rankings=rankings,
            metrics_comparison_table=df_metrics,
            resource_comparison_table=df_resources,
            summary_notes=summary_notes,
        )
