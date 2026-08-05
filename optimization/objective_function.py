"""
QuantLab Multi-Objective Fitness Evaluation Engine.

Evaluates single and multi-objective fitness functions combining Profit Factor, Sharpe Ratio,
Sortino Ratio, Calmar Ratio, Max Drawdown (penalty), Recovery Factor, Expectancy, Total Return,
and Execution Latency via scalarized weighted combination or Pareto dominance sorting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ObjectiveWeight:
    """Dataclass mapping metric key to optimization weight and direction."""

    metric_name: str
    weight: float = 1.0
    maximize: bool = True  # True to maximize, False to minimize (penalty)


class ObjectiveFunction:
    """Master Multi-Objective Fitness Evaluator."""

    def __init__(self, weights: Optional[List[ObjectiveWeight]] = None) -> None:
        """Initialize ObjectiveFunction.

        Args:
            weights: List of ObjectiveWeight definitions.
        """
        self.weights = weights or self._default_weights()

    @staticmethod
    def _default_weights() -> List[ObjectiveWeight]:
        """Get standard institutional default multi-objective weights."""
        return [
            ObjectiveWeight("sharpe_ratio", weight=0.25, maximize=True),
            ObjectiveWeight("profit_factor", weight=0.20, maximize=True),
            ObjectiveWeight("sortino_ratio", weight=0.15, maximize=True),
            ObjectiveWeight("max_drawdown", weight=0.15, maximize=False),  # Penalty
            ObjectiveWeight("recovery_factor", weight=0.10, maximize=True),
            ObjectiveWeight("expectancy", weight=0.10, maximize=True),
            ObjectiveWeight("execution_time_sec", weight=0.05, maximize=False),  # Latency penalty
        ]

    def _normalize_metric(self, name: str, value: float) -> float:
        """Normalize raw metric values to normalized [0, 100] scale.

        Args:
            name: Metric identifier.
            value: Raw numerical value.

        Returns:
            Normalized float value.
        """
        if name == "sharpe_ratio":
            return float(np.clip(value / 3.0 * 100.0, 0.0, 100.0))
        elif name == "sortino_ratio":
            return float(np.clip(value / 4.0 * 100.0, 0.0, 100.0))
        elif name == "profit_factor":
            return float(np.clip((value - 1.0) / 2.0 * 100.0, 0.0, 100.0))
        elif name == "max_drawdown":
            return float(np.clip(value / 50.0 * 100.0, 0.0, 100.0))
        elif name == "recovery_factor":
            return float(np.clip(value / 5.0 * 100.0, 0.0, 100.0))
        elif name == "calmar_ratio":
            return float(np.clip(value / 3.0 * 100.0, 0.0, 100.0))
        elif name == "expectancy":
            return float(np.clip(value / 100.0 * 100.0, 0.0, 100.0))
        elif name == "win_rate":
            return float(np.clip((value - 30.0) / 50.0 * 100.0, 0.0, 100.0))
        elif name == "execution_time_sec":
            return float(np.clip(value / 10.0 * 100.0, 0.0, 100.0))
        else:
            return float(np.clip(value, 0.0, 100.0))

    def evaluate(self, metrics: Dict[str, Any], execution_time_sec: float = 0.0) -> float:
        """Evaluate scalarized fitness score for a candidate solution (higher is better).

        Args:
            metrics: Calculated metric dictionary.
            execution_time_sec: Candidate execution time.

        Returns:
            Composite fitness score float.
        """
        data = {**metrics, "execution_time_sec": float(execution_time_sec)}
        total_fitness = 0.0
        total_weight = sum(w.weight for w in self.weights)

        if total_weight <= 0:
            return 0.0

        for obj in self.weights:
            raw_val = float(data.get(obj.metric_name, 0.0))
            norm_val = self._normalize_metric(obj.metric_name, raw_val)

            if not obj.maximize:
                norm_val = 100.0 - norm_val  # Invert penalty metrics

            total_fitness += norm_val * obj.weight

        composite_score = total_fitness / total_weight
        return float(round(composite_score, 4))
