"""
QuantLab Optimization Report Engine.

Generates formal institutional strategy optimization reports containing Executive Summary,
Configuration, Search Space Specifications, Algorithm Details, Top 10 Configurations Leaderboard,
Convergence Analysis, Conclusions, and Recommendations.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

from optimization.history import IterationRecord, OptimizationHistory
from optimization.logger import get_optimization_logger
from optimization.search_space import SearchSpace

logger = get_optimization_logger("ReportEngine")


@dataclass
class OptimizationReport:
    """Dataclass holding optimization report content."""

    title: str
    date: str
    executive_summary: str
    search_space_description: str
    algorithm_description: str
    top_configurations: List[Dict[str, Any]]
    conclusions: List[str]
    recommendation: str
    markdown_content: str

    def save_markdown(self, filepath: str) -> str:
        """Save report to Markdown file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.markdown_content)
        return filepath


class OptimizationReportEngine:
    """Master Institutional Optimization Report Generator."""

    def generate_report(
        self,
        strategy_name: str,
        algorithm_name: str,
        search_space: SearchSpace,
        history: OptimizationHistory,
    ) -> OptimizationReport:
        """Generate a complete formal institutional optimization report.

        Returns:
            OptimizationReport object.
        """
        all_recs = history.get_all_records()
        top_recs = history.get_top_solutions(k=10)
        best_rec = top_recs[0] if top_recs else None

        exec_summary = (
            f"Strategy hyperparameter optimization for '{strategy_name}' using algorithm '{algorithm_name}'. "
            f"Evaluated {len(all_recs)} total candidate parameter sets across {search_space.dimension} dimensions. "
            f"Best solution achieved a composite fitness score of {best_rec.fitness_score:.4f}."
            if best_rec
            else "No valid solutions evaluated."
        )

        space_desc = (
            f"Search Space: {search_space.name} (Dimension D={search_space.dimension})\n"
            f"Parameters: {list(search_space.flat_parameters.keys())}"
        )

        algo_desc = f"Optimization Algorithm: {algorithm_name}"

        top_configs = [
            {
                "rank": idx + 1,
                "eval_id": r.evaluation_id,
                "fitness": r.fitness_score,
                "parameters": r.parameters,
                "metrics": r.metrics,
            }
            for idx, r in enumerate(top_recs)
        ]

        if best_rec and best_rec.fitness_score >= 70.0:
            recommendation = "RECOMMENDED_FOR_RESEARCH_VALIDATION"
            rec_text = "RECOMMENDED. Top candidate parameters demonstrate high multi-objective fitness. Promote to Walk Forward & Research Engine validation."
        else:
            recommendation = "NEEDS_RE-OPTIMIZATION"
            rec_text = "NEEDS RE-OPTIMIZATION. Candidate parameters achieved insufficient fitness. Adjust search space bounds or objective weights."

        md_lines = [
            f"# QUANTLAB STRATEGY OPTIMIZATION REPORT",
            f"## Strategy: {strategy_name}",
            f"**Report Date**: {datetime.now(timezone.utc).isoformat()}",
            f"**Algorithm**: `{algorithm_name}` | **Evaluations**: `{len(all_recs)}`",
            "",
            "---",
            "### 1. EXECUTIVE SUMMARY",
            exec_summary,
            "",
            "### 2. SEARCH SPACE DEFINITION",
            "```text",
            space_desc,
            "```",
            "",
            "### 3. ALGORITHM SPECIFICATION",
            algo_desc,
            "",
            "### 4. TOP 10 CONFIGURATIONS LEADERBOARD",
            "| Rank | Eval ID | Fitness Score | Parameters |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for cfg in top_configs:
            md_lines.append(
                f"| `{cfg['rank']}` | `{cfg['eval_id']}` | `{cfg['fitness']:.4f}` | `{json.dumps(cfg['parameters'])}` |"
            )

        md_lines.extend(
            [
                "",
                "### 5. CONCLUSIONS & RECOMMENDATION",
                f"**Recommendation**: `{recommendation}`",
                rec_text,
                "",
                "---",
                "*QuantLab Optimization Engine - Institutional Report*",
            ]
        )

        full_md = "\n".join(md_lines)

        return OptimizationReport(
            title=f"Optimization Report - {strategy_name}",
            date=datetime.now(timezone.utc).isoformat(),
            executive_summary=exec_summary,
            search_space_description=space_desc,
            algorithm_description=algo_desc,
            top_configurations=top_configs,
            conclusions=[rec_text],
            recommendation=recommendation,
            markdown_content=full_md,
        )
