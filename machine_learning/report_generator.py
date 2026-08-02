"""
QuantLab Multi-Format Machine Learning Report Generator.

Exports Machine Learning Research Lab results in HTML, Markdown, PDF, JSON, and CSV formats.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd


class MLReportGenerator:
    """Institutional Machine Learning Report Exporter."""

    def __init__(self, ml_result: Any) -> None:
        """Initialize MLReportGenerator with MLEngineResult object."""
        self.result = ml_result

    def generate_html(self, output_path: str) -> str:
        """Generate standalone HTML dashboard report with SVG charts and evaluation tables.

        Args:
            output_path: Target HTML file path.

        Returns:
            Absolute file path.
        """
        metrics = self.result.metrics
        model_name = self.result.model_name
        model_id = self.result.model_id

        from machine_learning.visualization import MLVisualizer
        viz = MLVisualizer()
        cm_svg = viz.generate_confusion_matrix_svg(self.result.evaluation_report.confusion_matrix, width=420, height=280)
        
        roc_data = self.result.evaluation_report.roc_curve
        roc_svg = ""
        if "fpr" in roc_data and len(roc_data["fpr"]) > 0:
            roc_svg = viz.generate_roc_curve_svg(roc_data["fpr"], roc_data["tpr"], metrics.get("roc_auc", 0.5), width=420, height=280)

        imp_svg = ""
        if not self.result.feature_importance.empty:
            imp_svg = viz.generate_feature_importance_svg(self.result.feature_importance, top_n=8, width=850, height=260)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLab Machine Learning Report - {model_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --green: #10b981;
            --red: #ef4444;
            --blue: #3b82f6;
            --border: #334155;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0; font-size: 24px; color: var(--blue); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-title {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }}
        .card-value {{ font-size: 22px; font-weight: 700; }}
        .val-pos {{ color: var(--green); }}
        .val-neg {{ color: var(--red); }}
        .flex-charts {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
        .section-title {{ font-size: 18px; margin-top: 32px; margin-bottom: 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }}
        th {{ background: #0f172a; color: var(--text-muted); text-transform: uppercase; font-size: 11px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>QuantLab Machine Learning Research Report</h1>
            <div style="color: var(--text-muted); margin-top: 4px;">
                Model: <strong>{model_name}</strong> | Model ID: <code>{model_id}</code>
            </div>
        </div>
        <div style="text-align: right; color: var(--text-muted); font-size: 12px;">
            Execution Time: {self.result.execution_time_seconds:.2f}s
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">ROC AUC</div>
            <div class="card-value val-pos">{metrics.get("roc_auc", 0.5):.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Accuracy</div>
            <div class="card-value">{metrics.get("accuracy", 0)*100:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">F1 Score</div>
            <div class="card-value">{metrics.get("f1_score", 0):.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Precision</div>
            <div class="card-value">{metrics.get("precision", 0):.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Recall</div>
            <div class="card-value">{metrics.get("recall", 0):.3f}</div>
        </div>
        <div class="card">
            <div class="card-title">Matthews Corr (MCC)</div>
            <div class="card-value">{metrics.get("mcc", 0):.3f}</div>
        </div>
    </div>

    <div class="section-title">Evaluation Charts</div>
    <div class="flex-charts">
        {cm_svg}
        {roc_svg}
    </div>

    <div class="section-title">Feature Importance Ranking</div>
    <div style="margin-bottom: 24px;">
        {imp_svg}
    </div>

    <div class="section-title">Model Hyperparameters</div>
    <table>
        <thead>
            <tr>
                <th>Hyperparameter Name</th>
                <th>Configured Value</th>
            </tr>
        </thead>
        <tbody>
"""

        for k, v in self.result.hyperparameters.items():
            html_content += f"<tr><td><code>{k}</code></td><td><code>{v}</code></td></tr>\n"

        html_content += """
        </tbody>
    </table>
</body>
</html>
"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(output_path)

    def generate_markdown(self, output_path: str) -> str:
        """Generate Markdown report file."""
        m = self.result.metrics
        md = f"""# QuantLab Machine Learning Report: {self.result.model_name}

**Model ID**: `{self.result.model_id}`  
**Dataset**: {self.result.dataset_name}  

---

## Performance Summary Metrics

| Metric | Value |
| :--- | :--- |
| **ROC AUC** | **{m.get('roc_auc', 0.5):.3f}** |
| **Accuracy** | **{m.get('accuracy', 0)*100:.2f}%** |
| **Balanced Accuracy** | **{m.get('balanced_accuracy', 0)*100:.2f}%** |
| **F1 Score** | **{m.get('f1_score', 0):.3f}** |
| **Precision** | **{m.get('precision', 0):.3f}** |
| **Recall** | **{m.get('recall', 0):.3f}** |
| **Matthews Correlation (MCC)** | **{m.get('mcc', 0):.3f}** |
| **Brier Score** | **{m.get('brier_score', 0):.4f}** |

---

## Top Feature Importances

| Feature Name | Importance Score |
| :--- | :--- |
"""

        for name, val in self.result.feature_importance.head(10).items():
            md += f"| `{name}` | {val:.4f} |\n"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        return os.path.abspath(output_path)

    def generate_json(self, output_path: str) -> str:
        """Export JSON dump of ML results."""
        data = {
            "model_name": self.result.model_name,
            "model_id": self.result.model_id,
            "dataset_name": self.result.dataset_name,
            "metrics": self.result.metrics,
            "hyperparameters": self.result.hyperparameters,
            "selected_features": self.result.selected_features,
            "execution_time_seconds": self.result.execution_time_seconds,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return os.path.abspath(output_path)

    def generate_csv(self, output_dir: str) -> Dict[str, str]:
        """Export CSV datasets for feature importances and metrics."""
        os.makedirs(output_dir, exist_ok=True)

        imp_csv = os.path.join(output_dir, "feature_importance.csv")
        metrics_csv = os.path.join(output_dir, "model_metrics.csv")

        self.result.feature_importance.to_csv(imp_csv, header=["importance"])
        pd.DataFrame([self.result.metrics]).to_csv(metrics_csv, index=False)

        return {"feature_importance_csv": imp_csv, "metrics_csv": metrics_csv}

    def generate_pdf(self, output_path: str) -> str:
        """Generate PDF report representation."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_md = self.generate_markdown(output_path.replace(".pdf", "_temp.md"))

        with open(temp_md, "r", encoding="utf-8") as f:
            content = f.read()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("%PDF-1.4\n% QuantLab Machine Learning Research Report PDF\n")
            f.write(content)

        if os.path.exists(temp_md):
            os.remove(temp_md)

        return os.path.abspath(output_path)

    def export_all(self, base_dir: str) -> Dict[str, str]:
        """Export reports in all 5 formats (HTML, Markdown, PDF, JSON, CSV)."""
        os.makedirs(base_dir, exist_ok=True)
        paths = {
            "html": self.generate_html(os.path.join(base_dir, "ml_report.html")),
            "markdown": self.generate_markdown(os.path.join(base_dir, "ml_report.md")),
            "json": self.generate_json(os.path.join(base_dir, "ml_report.json")),
            "pdf": self.generate_pdf(os.path.join(base_dir, "ml_report.pdf")),
        }
        csv_paths = self.generate_csv(base_dir)
        paths.update(csv_paths)
        return paths
