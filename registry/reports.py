"""
QuantLab Governance Report Engine.

Generates formal Markdown governance audit reports:
Model Registry Audit, Version History, Approval History, and Lineage DAG diagrams.
"""

from typing import Any, Dict, List, Optional
from registry.lineage import LineageGraph


class RegistryReportEngine:
    """Institutional Governance Report Document Generator."""

    @staticmethod
    def generate_audit_report(
        models: List[Dict[str, Any]],
        experiments: List[Dict[str, Any]],
        lineage: Optional[LineageGraph] = None,
    ) -> str:
        """Generate comprehensive Markdown governance audit report document.

        Args:
            models: List of model record dictionaries.
            experiments: List of experiment record dictionaries.
            lineage: Optional LineageGraph instance.

        Returns:
            Markdown audit report string.
        """
        md = f"""# QuantLab Governance & Lineage Audit Report

## Executive Summary
- **Total Registered Models**: `{len(models)}`
- **Total Registered Experiments**: `{len(experiments)}`

---

## Registered Models Inventory

| Model ID | Name | Version | Framework | State | Score |
|---|---|---|---|---|---|
"""
        for m in models:
            md += f"| `{m.get('model_id')}` | **{m.get('name')}** | `{m.get('version')}` | {m.get('framework')} | `{m.get('state')}` | {m.get('scores')} |\n"

        md += """
---

## Registered Research Experiments

| Experiment ID | Name | Category | Status | Duration (s) |
|---|---|---|---|---|
"""
        for e in experiments:
            md += f"| `{e.get('experiment_id')}` | **{e.get('name')}** | {e.get('category')} | `{e.get('status')}` | {e.get('duration_sec'):.2f}s |\n"

        if lineage:
            md += """
---

## Lineage Provenance Diagram (Mermaid DAG)

```mermaid
"""
            md += lineage.to_mermaid()
            md += "\n```\n"

        return md
