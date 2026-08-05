"""
QuantLab Scientific Research Report Generator Engine.

Generates formal institutional quantitative research reports containing Executive Summary,
Objective, Configuration, Methodology, Metric Results, Benchmarks, Conclusions, and Formal Recommendation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from research.comparator import ComparisonResult
from research.experiment import Experiment, ExperimentStatus
from research.logger import get_research_logger
from research.scorer import ScoreResult, Scorer
from research.validator import ValidationResult, Validator

logger = get_research_logger("ReportEngine")


@dataclass
class ResearchReport:
    """Dataclass holding complete institutional research document payload."""

    title: str
    experiment_uuid: str
    date: str
    executive_summary: str
    objective: str
    configuration_summary: str
    methodology: str
    metrics_summary: Dict[str, Any]
    validation_summary: Optional[Dict[str, Any]]
    score_summary: Optional[Dict[str, Any]]
    comparison_summary: Optional[Dict[str, Any]]
    conclusions: List[str]
    recommendation: str  # 'APPROVED_FOR_PRODUCTION', 'REJECTED', 'NEEDS_REVISION'
    status: str
    markdown_content: str

    def save_markdown(self, filepath: str) -> str:
        """Save report content to Markdown file.

        Returns:
            Absolute file path.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.markdown_content)
        return filepath


class ReportEngine:
    """Institutional Report Generator Engine for QuantLab Scientific Research."""

    def __init__(
        self,
        validator: Optional[Validator] = None,
        scorer: Optional[Scorer] = None,
    ) -> None:
        """Initialize ReportEngine.

        Args:
            validator: Optional Validator instance.
            scorer: Optional Scorer instance.
        """
        self.validator = validator or Validator()
        self.scorer = scorer or Scorer()

    def generate_report(
        self,
        experiment: Experiment,
        comparison_result: Optional[ComparisonResult] = None,
    ) -> ResearchReport:
        """Generate a complete, formal institutional scientific research report.

        Args:
            experiment: Target Experiment object.
            comparison_result: Optional comparison outputs across candidate experiments.

        Returns:
            ResearchReport object.
        """
        logger.info(f"Generating research report for experiment UUID={experiment.uuid}...")

        # 1. Run validation and scoring
        val_res: ValidationResult = self.validator.validate(experiment)
        score_res: ScoreResult = self.scorer.score(experiment)

        # 2. Executive Summary
        exec_summary = (
            f"Scientific research evaluation of strategy '{experiment.name}' (Version {experiment.version}). "
            f"Execution completed in {experiment.execution_time:.4f} seconds with status '{experiment.status}'. "
            f"Achieved an Overall Score of {score_res.overall_score}/100 (Grade: {score_res.grade}) "
            f"and validation outcome '{val_res.status}'."
        )

        # 3. Objective & Configuration
        objective = (
            f"Evaluate the quantitative risk-adjusted performance, drawdown stability, "
            f"and statistical robustness of strategy '{experiment.name}' on asset '{experiment.asset}' "
            f"({experiment.timeframe} timeframe) operating with broker '{experiment.broker}'."
        )

        config_summary = (
            f"Asset: {experiment.asset} | Timeframe: {experiment.timeframe} | Broker: {experiment.broker}\n"
            f"Random Seed: {experiment.random_seed} | Parameters: {experiment.parameters}"
        )

        # 4. Methodology
        methodology = (
            "Multi-stage institutional validation framework combining realistic Backtesting, "
            "Walk Forward Out-Of-Sample optimization, Monte Carlo trade permutation simulations, "
            "and multi-criteria risk validation."
        )

        # 5. Recommendation logic
        if val_res.status == "REJECTED" or score_res.overall_score < 60.0:
            recommendation = "REJECTED"
            rec_text = "DO NOT DEPLOY. Strategy failed institutional risk validation bounds or achieved insufficient score."
        elif score_res.overall_score < 75.0:
            recommendation = "NEEDS_REVISION"
            rec_text = "NEEDS REVISION. Strategy passed initial checks but requires parameter refinement and further robustness testing."
        else:
            recommendation = "APPROVED_FOR_PRODUCTION"
            rec_text = "APPROVED FOR MODEL REGISTRY & SENTINEL PROMOTION. Strategy satisfies all institutional safety, performance, and robustness standards."

        # 6. Build Markdown layout
        md_lines = [
            f"# QUANTLAB SCIENTIFIC RESEARCH REPORT",
            f"## Experiment: {experiment.name} (v{experiment.version})",
            f"**Report Date**: {datetime.now(timezone.utc).isoformat()}",
            f"**Experiment UUID**: `{experiment.uuid}`",
            f"**Author**: {experiment.author}",
            "",
            "---",
            "### 1. EXECUTIVE SUMMARY",
            exec_summary,
            "",
            "### 2. RESEARCH OBJECTIVE",
            objective,
            "",
            "### 3. CONFIGURATION & PARAMETERS",
            "```text",
            config_summary,
            "```",
            "",
            "### 4. METHODOLOGY",
            methodology,
            "",
            "### 5. INSTITUTIONAL METRICS SUMMARY",
            "| Metric | Observed Value |",
            "| :--- | :--- |",
        ]

        for k, v in experiment.results.items():
            if isinstance(v, (int, float, str)):
                md_lines.append(f"| `{k}` | `{v}` |")

        md_lines.extend(
            [
                "",
                "### 6. VALIDATION & SCORING OUTCOME",
                f"- **Validation Status**: `{val_res.status}` ({len(val_res.passed_rules)} passed, {len(val_res.failed_rules)} failed)",
                f"- **Overall Score**: `{score_res.overall_score}/100` (Grade `{score_res.grade}`)",
                f"- **Institutional Risk Score**: `{score_res.institutional_score}/100`",
                f"- **Robustness Score**: `{score_res.robustness_score}/100`",
                "",
            ]
        )

        if comparison_result:
            md_lines.extend(
                [
                    "### 7. CANDIDATE BENCHMARK COMPARISON",
                    f"Compared across `{comparison_result.experiments_count}` candidate experiments.",
                    f"**Winning Strategy**: `{comparison_result.winner_name}` (UUID: `{comparison_result.winner_uuid}`)",
                    "",
                ]
            )

        md_lines.extend(
            [
                "### 8. CONCLUSIONS & FINAL RECOMMENDATION",
                f"**Recommendation**: `{recommendation}`",
                rec_text,
                "",
                "---",
                "*QuantLab Scientific Research Engine - Institutional Report*",
            ]
        )

        full_md = "\n".join(md_lines)

        return ResearchReport(
            title=f"Research Report - {experiment.name}",
            experiment_uuid=experiment.uuid,
            date=datetime.now(timezone.utc).isoformat(),
            executive_summary=exec_summary,
            objective=objective,
            configuration_summary=config_summary,
            methodology=methodology,
            metrics_summary=experiment.results,
            validation_summary={
                "status": val_res.status,
                "passed": len(val_res.passed_rules),
                "failed": len(val_res.failed_rules),
            },
            score_summary={
                "overall": score_res.overall_score,
                "institutional": score_res.institutional_score,
                "robustness": score_res.robustness_score,
                "grade": score_res.grade,
            },
            comparison_summary={
                "winner_uuid": comparison_result.winner_uuid if comparison_result else experiment.uuid,
            }
            if comparison_result
            else None,
            conclusions=[rec_text],
            recommendation=recommendation,
            status=str(experiment.status),
            markdown_content=full_md,
        )
