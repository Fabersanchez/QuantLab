"""
QuantLab Institutional Experiment Scorer.

Calculates composite scientific quality ratings (0 to 100) combining Overall Score,
Institutional Score (risk-adjusted quality), and Robustness Score (stability & stress resilience).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from research.experiment import Experiment
from research.logger import get_research_logger

logger = get_research_logger("Scorer")


@dataclass
class ScoreWeights:
    """Dataclass holding configurable metric weights for scoring calculations."""

    profit_factor: float = 0.15
    sharpe_ratio: float = 0.20
    sortino_ratio: float = 0.15
    calmar_ratio: float = 0.10
    recovery_factor: float = 0.10
    drawdown: float = 0.10  # Max drawdown penalty
    expectancy: float = 0.10
    win_rate: float = 0.05
    risk_reward: float = 0.05


@dataclass
class ScoreResult:
    """Dataclass containing institutional score evaluation output."""

    experiment_uuid: str
    overall_score: float
    institutional_score: float
    robustness_score: float
    grade: str  # 'S', 'A', 'B', 'C', 'D', 'F'
    breakdown: Dict[str, float]


class Scorer:
    """Master Institutional Scoring Engine for QuantLab research experiments."""

    def __init__(self, weights: Optional[ScoreWeights] = None) -> None:
        """Initialize Scorer with metric weights.

        Args:
            weights: Optional ScoreWeights instance.
        """
        self.weights = weights or ScoreWeights()

    def _extract_metric(self, exp: Experiment, key: str) -> float:
        """Helper to extract metric value safely from experiment.

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

    @staticmethod
    def _normalize_metric(val: float, min_b: float, max_b: float) -> float:
        """Normalize metric value to 0-100 scale within bounds.

        Args:
            val: Observed value.
            min_b: Minimum benchmark boundary.
            max_b: Maximum benchmark boundary.

        Returns:
            Score from 0.0 to 100.0.
        """
        if max_b <= min_b:
            return 50.0
        clipped = max(min_b, min(max_b, val))
        return ((clipped - min_b) / (max_b - min_b)) * 100.0

    def score(self, experiment: Experiment) -> ScoreResult:
        """Calculate Overall Score, Institutional Score, and Robustness Score for an experiment.

        Args:
            experiment: Target experiment instance.

        Returns:
            ScoreResult object containing scores and grade.
        """
        # Extract metrics
        pf = self._extract_metric(experiment, "profit_factor")
        sharpe = self._extract_metric(experiment, "sharpe_ratio")
        sortino = self._extract_metric(experiment, "sortino_ratio")
        calmar = self._extract_metric(experiment, "calmar_ratio")
        rec = self._extract_metric(experiment, "recovery_factor")
        dd = self._extract_metric(experiment, "max_drawdown")
        exp_val = self._extract_metric(experiment, "expectancy")
        win_rate = self._extract_metric(experiment, "win_rate")
        rr = self._extract_metric(experiment, "risk_reward")

        # Extract Walk Forward & Monte Carlo metrics if present
        res = experiment.results or {}
        wf_efficiency = float(res.get("walk_forward_efficiency", res.get("efficiency", 0.5)))
        mc_ruin_prob = float(res.get("monte_carlo_ruin_prob", res.get("ruin_probability", 0.0)))

        # Compute normalized sub-scores (0-100)
        s_pf = self._normalize_metric(pf, 0.5, 3.0)
        s_sharpe = self._normalize_metric(sharpe, 0.0, 3.0)
        s_sortino = self._normalize_metric(sortino, 0.0, 4.0)
        s_calmar = self._normalize_metric(calmar, 0.0, 3.0)
        s_rec = self._normalize_metric(rec, 0.0, 5.0)
        s_dd = 100.0 - self._normalize_metric(dd, 0.0, 50.0)  # Inverted for drawdown
        s_exp = self._normalize_metric(exp_val, 0.0, 100.0)
        s_win = self._normalize_metric(win_rate, 30.0, 80.0)
        s_rr = self._normalize_metric(rr, 0.5, 3.0)

        # 1. Overall Score
        w = self.weights
        overall = (
            s_pf * w.profit_factor
            + s_sharpe * w.sharpe_ratio
            + s_sortino * w.sortino_ratio
            + s_calmar * w.calmar_ratio
            + s_rec * w.recovery_factor
            + s_dd * w.drawdown
            + s_exp * w.expectancy
            + s_win * w.win_rate
            + s_rr * w.risk_reward
        )
        overall = round(max(0.0, min(100.0, overall)), 2)

        # 2. Institutional Score (focused on risk-adjusted ratios & drawdowns)
        institutional = (s_sharpe * 0.30) + (s_sortino * 0.25) + (s_calmar * 0.20) + (s_dd * 0.25)
        institutional = round(max(0.0, min(100.0, institutional)), 2)

        # 3. Robustness Score (combines WF efficiency, MC ruin protection, and SQN)
        s_wf = self._normalize_metric(wf_efficiency, 0.0, 1.0)
        s_mc = 100.0 - self._normalize_metric(mc_ruin_prob, 0.0, 0.2)
        sqn = self._extract_metric(experiment, "sqn")
        s_sqn = self._normalize_metric(sqn, 0.0, 5.0)

        robustness = (s_wf * 0.40) + (s_mc * 0.40) + (s_sqn * 0.20)
        robustness = round(max(0.0, min(100.0, robustness)), 2)

        # Determine letter grade
        if overall >= 90.0:
            grade = "S"
        elif overall >= 80.0:
            grade = "A"
        elif overall >= 70.0:
            grade = "B"
        elif overall >= 60.0:
            grade = "C"
        elif overall >= 50.0:
            grade = "D"
        else:
            grade = "F"

        breakdown = {
            "profit_factor_score": round(s_pf, 2),
            "sharpe_score": round(s_sharpe, 2),
            "sortino_score": round(s_sortino, 2),
            "calmar_score": round(s_calmar, 2),
            "drawdown_score": round(s_dd, 2),
            "expectancy_score": round(s_exp, 2),
        }

        logger.info(
            f"Scored experiment UUID={experiment.uuid}: Overall={overall}/100, "
            f"Institutional={institutional}/100, Robustness={robustness}/100, Grade={grade}"
        )

        return ScoreResult(
            experiment_uuid=experiment.uuid,
            overall_score=overall,
            institutional_score=institutional,
            robustness_score=robustness,
            grade=grade,
            breakdown=breakdown,
        )
