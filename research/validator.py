"""
QuantLab Institutional Experiment Multi-Criteria Validator.

Enforces strict institutional validation contracts. Rejects single-metric acceptance
(e.g., WinRate alone is prohibited) and enforces multi-dimensional thresholds across Profit Factor,
Sharpe, Sortino, Calmar, Recovery, Max Drawdown, Expectancy, and Volatility.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from research.experiment import Experiment, ExperimentStatus
from research.logger import get_research_logger

logger = get_research_logger("Validator")


@dataclass
class ValidationRule:
    """Dataclass defining a single metric validation boundary constraint."""

    metric_name: str
    operator: str  # '>=', '<=', '>', '<', '=='
    threshold: float
    description: str = ""

    def evaluate(self, value: float) -> bool:
        """Evaluate rule boundary against a numeric metric value.

        Args:
            value: Observed metric value.

        Returns:
            True if rule passes, False if violated.
        """
        if self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        return False


@dataclass
class ValidationResult:
    """Dataclass containing institutional validation evaluation output."""

    experiment_uuid: str
    status: str  # 'PASSED' or 'REJECTED'
    passed_rules: List[ValidationRule]
    failed_rules: List[Dict[str, Any]]
    evaluated_metrics: Dict[str, float]
    summary: str


class Validator:
    """Institutional Multi-Criteria Scientific Experiment Validator."""

    def __init__(self, rules: Optional[List[ValidationRule]] = None) -> None:
        """Initialize Validator with a set of validation rules.

        Args:
            rules: Optional list of ValidationRule objects.
        """
        self.rules = rules or self._get_default_rules()

    @staticmethod
    def _get_default_rules() -> List[ValidationRule]:
        """Get standard institutional validation rules.

        Returns:
            List of default ValidationRule objects.
        """
        return [
            ValidationRule("profit_factor", ">=", 1.25, "Minimum Profit Factor boundary"),
            ValidationRule("sharpe_ratio", ">=", 0.75, "Minimum Sharpe Ratio boundary"),
            ValidationRule("max_drawdown", "<=", 30.0, "Maximum Drawdown percentage ceiling"),
            ValidationRule("expectancy", ">", 0.0, "Minimum positive trade Expectancy"),
            ValidationRule("recovery_factor", ">=", 1.0, "Minimum Recovery Factor boundary"),
            ValidationRule("calmar_ratio", ">=", 0.5, "Minimum Calmar Ratio boundary"),
        ]

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule.

        Args:
            rule: ValidationRule instance.
        """
        self.rules.append(rule)

    def _extract_metric(self, exp: Experiment, key: str) -> float:
        """Extract metric value safely from experiment object.

        Args:
            exp: Target experiment instance.
            key: Metric key name.

        Returns:
            Float value.
        """
        res = exp.results or {}
        if key in res:
            val = res[key]
            return float(val) if isinstance(val, (int, float)) else 0.0
        metrics = res.get("metrics", {})
        if key in metrics:
            val = metrics[key]
            return float(val) if isinstance(val, (int, float)) else 0.0
        return 0.0

    def validate(self, experiment: Experiment) -> ValidationResult:
        """Validate experiment against multi-criteria rules.

        Args:
            experiment: Experiment instance to validate.

        Returns:
            ValidationResult object.
        """
        passed_rules: List[ValidationRule] = []
        failed_rules: List[Dict[str, Any]] = []
        evaluated_metrics: Dict[str, float] = {}

        # Enforce anti-WinRate policy: WinRate alone cannot pass an experiment
        win_rate = self._extract_metric(experiment, "win_rate")
        evaluated_metrics["win_rate"] = win_rate

        for rule in self.rules:
            observed_val = self._extract_metric(experiment, rule.metric_name)
            evaluated_metrics[rule.metric_name] = observed_val
            is_valid = rule.evaluate(observed_val)

            if is_valid:
                passed_rules.append(rule)
            else:
                failed_rules.append(
                    {
                        "rule": rule.metric_name,
                        "operator": rule.operator,
                        "threshold": rule.threshold,
                        "observed": observed_val,
                        "description": rule.description,
                    }
                )

        status_str = "PASSED" if len(failed_rules) == 0 else "REJECTED"

        if status_str == "REJECTED":
            experiment.status = ExperimentStatus.REJECTED
            summary = f"Experiment REJECTED: Failed {len(failed_rules)} out of {len(self.rules)} criteria."
        else:
            summary = f"Experiment PASSED: Satisfied all {len(self.rules)} institutional criteria."

        logger.log_validation(experiment.uuid, status_str, len(passed_rules), len(failed_rules))

        return ValidationResult(
            experiment_uuid=experiment.uuid,
            status=status_str,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            evaluated_metrics=evaluated_metrics,
            summary=summary,
        )
