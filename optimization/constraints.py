"""
QuantLab Optimization Constraints Engine.

Enforces boundary constraints on candidate solutions: maximum drawdown, minimum profit factor,
minimum Sharpe ratio, minimum Sortino ratio, minimum recovery factor, maximum execution time,
maximum RAM MB, maximum CPU usage %, and minimum trade count.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class ConstraintRule:
    """Dataclass defining a single metric constraint boundary."""

    name: str
    operator: str  # '<=', '>=', '<', '>', '=='
    threshold: float
    description: str = ""

    def evaluate(self, value: float) -> bool:
        """Evaluate constraint boundary rule against observed metric value."""
        if self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        return False


class OptimizationConstraints:
    """Master Manager for Optimization Candidate Constraints."""

    def __init__(
        self,
        max_drawdown: Optional[float] = None,
        min_profit_factor: Optional[float] = None,
        min_sharpe: Optional[float] = None,
        min_sortino: Optional[float] = None,
        min_recovery: Optional[float] = None,
        max_time_sec: Optional[float] = None,
        max_ram_mb: Optional[float] = None,
        max_cpu_pct: Optional[float] = None,
        min_trades: Optional[int] = None,
    ) -> None:
        """Initialize OptimizationConstraints.

        Args:
            max_drawdown: Max drawdown % ceiling (e.g., 25.0).
            min_profit_factor: Min Profit Factor floor (e.g., 1.5).
            min_sharpe: Min Sharpe Ratio floor (e.g., 1.0).
            min_sortino: Min Sortino Ratio floor (e.g., 1.0).
            min_recovery: Min Recovery Factor floor (e.g., 2.0).
            max_time_sec: Max evaluation time seconds.
            max_ram_mb: Max RAM peak consumption MB.
            max_cpu_pct: Max CPU usage %.
            min_trades: Min required trade count.
        """
        self.rules: List[ConstraintRule] = []
        self.custom_constraints: List[Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]] = []

        if max_drawdown is not None:
            self.rules.append(ConstraintRule("max_drawdown", "<=", max_drawdown, f"Drawdown <= {max_drawdown}%"))
        if min_profit_factor is not None:
            self.rules.append(
                ConstraintRule("profit_factor", ">=", min_profit_factor, f"Profit Factor >= {min_profit_factor}")
            )
        if min_sharpe is not None:
            self.rules.append(ConstraintRule("sharpe_ratio", ">=", min_sharpe, f"Sharpe Ratio >= {min_sharpe}"))
        if min_sortino is not None:
            self.rules.append(ConstraintRule("sortino_ratio", ">=", min_sortino, f"Sortino Ratio >= {min_sortino}"))
        if min_recovery is not None:
            self.rules.append(ConstraintRule("recovery_factor", ">=", min_recovery, f"Recovery Factor >= {min_recovery}"))
        if max_time_sec is not None:
            self.rules.append(ConstraintRule("execution_time_sec", "<=", max_time_sec, f"Execution Time <= {max_time_sec}s"))
        if max_ram_mb is not None:
            self.rules.append(ConstraintRule("ram_peak_mb", "<=", max_ram_mb, f"RAM <= {max_ram_mb} MB"))
        if max_cpu_pct is not None:
            self.rules.append(ConstraintRule("cpu_usage_pct", "<=", max_cpu_pct, f"CPU <= {max_cpu_pct}%"))
        if min_trades is not None:
            self.rules.append(ConstraintRule("total_trades", ">=", float(min_trades), f"Trades >= {min_trades}"))

    def add_rule(self, rule: ConstraintRule) -> None:
        """Add custom ConstraintRule."""
        self.rules.append(rule)

    def add_custom_constraint(
        self, fn: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]]
    ) -> None:
        """Add custom constraint function accepting (metrics, execution_info) returning (is_valid, reason)."""
        self.custom_constraints.append(fn)

    def evaluate(
        self, metrics: Dict[str, Any], execution_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Evaluate metrics and execution resource usage against all rules.

        Args:
            metrics: Performance metrics dictionary.
            execution_info: Optional hardware/timing resource dictionary.

        Returns:
            Tuple of (is_valid_bool, list_of_violation_strings).
        """
        combined = {**metrics, **(execution_info or {})}
        violations: List[str] = []

        for rule in self.rules:
            if rule.name in combined:
                val = float(combined[rule.name])
                if not rule.evaluate(val):
                    violations.append(
                        f"Violated '{rule.name}': Observed {val:.4f}, expected {rule.operator} {rule.threshold:.4f} ({rule.description})"
                    )

        for fn in self.custom_constraints:
            try:
                valid, reason = fn(metrics, execution_info or {})
                if not valid:
                    violations.append(f"Violated Custom Constraint: {reason}")
            except Exception as e:
                violations.append(f"Custom Constraint Exception: {str(e)}")

        is_valid = len(violations) == 0
        return is_valid, violations
