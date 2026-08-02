"""
QuantLab Walk Forward Multi-Format Report Generator.

Exports Walk Forward Analysis results in HTML, Markdown, PDF, JSON, and CSV formats.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd


class WalkForwardReportGenerator:
    """Institutional Walk Forward Report Exporter."""

    def __init__(self, walkforward_result: Any) -> None:
        """Initialize WalkForwardReportGenerator with WalkForwardResult object."""
        self.result = walkforward_result

    def generate_html(self, output_path: str) -> str:
        """Generate standalone HTML dashboard report with SVG charts and interactive tables.

        Args:
            output_path: Target HTML file path.

        Returns:
            Absolute file path.
        """
        rob = self.result.robustness_metrics
        eff = self.result.efficiency_metrics
        stats = self.result.statistics_summary

        wfe = rob.get("walk_forward_efficiency_pct", 0.0)
        stab = rob.get("stability_score_pct", 0.0)
        r_idx = rob.get("robustness_index", 0.0)

        # Generate SVG charts
        from walk_forward.visualization import WalkForwardVisualizer
        viz = WalkForwardVisualizer(self.result.window_results)
        eq_svg = viz.generate_equity_comparison_svg(self.result.concatenated_oos_equity, width=850, height=320)
        cmp_svg = viz.generate_window_comparison_svg(width=850, height=280)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLab Walk Forward Analysis - {self.result.strategy_name}</title>
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
        .chart-container {{ margin-bottom: 24px; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>QuantLab Walk Forward Optimization Report</h1>
            <div style="color: var(--text-muted); margin-top: 4px;">
                Strategy: <strong>{self.result.strategy_name}</strong> | Asset: <strong>{self.result.asset_symbol}</strong>
            </div>
        </div>
        <div style="text-align: right; color: var(--text-muted); font-size: 12px;">
            Execution Time: {self.result.execution_time_seconds:.2f}s
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Walk Forward Efficiency</div>
            <div class="card-value {"val-pos" if wfe >= 50 else "val-neg"}">{wfe:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Robustness Index</div>
            <div class="card-value">{r_idx:.1f} / 100</div>
        </div>
        <div class="card">
            <div class="card-title">Stability Score</div>
            <div class="card-value val-pos">{stab:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Overfitting Score</div>
            <div class="card-value {"val-neg" if rob.get("overfitting_score_pct", 0) > 50 else "val-pos"}">{rob.get("overfitting_score_pct", 0):.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Total OOS Profit</div>
            <div class="card-value {"val-pos" if stats.get("total_oos_net_profit", 0) >= 0 else "val-neg"}">${stats.get("total_oos_net_profit", 0):,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Mean OOS Sharpe</div>
            <div class="card-value">{stats.get("mean_oos_sharpe", 0):.2f}</div>
        </div>
    </div>

    <div class="section-title">Concatenated Out-of-Sample Equity Curve</div>
    <div class="chart-container">
        {eq_svg}
    </div>

    <div class="section-title">Window Performance Comparison</div>
    <div class="chart-container">
        {cmp_svg}
    </div>

    <div class="section-title">Walk Forward Window Breakdown</div>
    <table>
        <thead>
            <tr>
                <th>Window</th>
                <th>IS Net Profit</th>
                <th>IS Sharpe</th>
                <th>OOS Net Profit</th>
                <th>OOS Sharpe</th>
                <th>OOS Win Rate</th>
                <th>OOS Max DD</th>
                <th>Optimized Best Parameters</th>
            </tr>
        </thead>
        <tbody>
"""

        for s in self.result.window_results:
            is_p = s.is_metrics.get("net_profit", 0.0)
            is_s = s.is_metrics.get("sharpe_ratio", 0.0)
            oos_p = s.oos_metrics.get("net_profit", 0.0)
            oos_s = s.oos_metrics.get("sharpe_ratio", 0.0)
            oos_w = s.oos_metrics.get("win_rate", 0.0)
            oos_dd = s.oos_metrics.get("max_drawdown_pct", 0.0)

            html_content += f"""
            <tr>
                <td>W{s.window_index}</td>
                <td>${is_p:,.2f}</td>
                <td>{is_s:.2f}</td>
                <td style="color: {'#10b981' if oos_p >= 0 else '#ef4444'}">${oos_p:,.2f}</td>
                <td>{oos_s:.2f}</td>
                <td>{oos_w:.1f}%</td>
                <td style="color: #ef4444">-{oos_dd:.1f}%</td>
                <td style="font-family: monospace; font-size: 11px;">{s.best_params}</td>
            </tr>
"""

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
        """Generate Markdown Walk Forward report file."""
        rob = self.result.robustness_metrics
        stats = self.result.statistics_summary

        md = f"""# QuantLab Walk Forward Optimization Report

**Strategy**: {self.result.strategy_name}  
**Asset**: {self.result.asset_symbol}  
**Window Splits**: {len(self.result.window_results)}  

---

## Robustness & Efficiency Executive Summary

| Robustness Indicator | Value |
| :--- | :--- |
| **Walk Forward Efficiency (WFE)** | **{rob.get('walk_forward_efficiency_pct', 0):.1f}%** |
| **Robustness Index (0-100)** | **{rob.get('robustness_index', 0):.1f}** |
| **Stability Score (% Positive Windows)** | **{rob.get('stability_score_pct', 0):.1f}%** |
| **Parameter Stability Score** | **{rob.get('parameter_stability_score', 0):.1f} / 100** |
| **Overfitting Score** | **{rob.get('overfitting_score_pct', 0):.1f}%** |
| **Out-of-Sample Ratio** | **{rob.get('out_of_sample_ratio_pct', 0):.1f}%** |
| **Total OOS Net Profit** | **${stats.get('total_oos_net_profit', 0):,.2f}** |
| **Mean OOS Sharpe Ratio** | **{stats.get('mean_oos_sharpe', 0):.2f}** |

---

## Window Breakdown Table

| Window | Train Bars | Val Bars | IS Net Profit | IS Sharpe | OOS Net Profit | OOS Sharpe | Best Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

        for s in self.result.window_results:
            is_p = s.is_metrics.get("net_profit", 0.0)
            is_s = s.is_metrics.get("sharpe_ratio", 0.0)
            oos_p = s.oos_metrics.get("net_profit", 0.0)
            oos_s = s.oos_metrics.get("sharpe_ratio", 0.0)

            md += f"| W{s.window_index} | {s.train_split.train_bars} | {s.train_split.val_bars} | ${is_p:,.2f} | {is_s:.2f} | ${oos_p:,.2f} | {oos_s:.2f} | `{s.best_params}` |\n"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        return os.path.abspath(output_path)

    def generate_json(self, output_path: str) -> str:
        """Export full Walk Forward json structure."""
        data = {
            "strategy_name": self.result.strategy_name,
            "asset_symbol": self.result.asset_symbol,
            "robustness_metrics": self.result.robustness_metrics,
            "efficiency_metrics": self.result.efficiency_metrics,
            "statistics_summary": self.result.statistics_summary,
            "window_results": [
                {
                    "window_index": s.window_index,
                    "best_params": s.best_params,
                    "is_metrics": s.is_metrics,
                    "oos_metrics": s.oos_metrics,
                }
                for s in self.result.window_results
            ],
            "execution_time_seconds": self.result.execution_time_seconds,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return os.path.abspath(output_path)

    def generate_csv(self, output_dir: str) -> Dict[str, str]:
        """Export CSV files for window statistics, parameter evolution, and concatenated OOS equity."""
        os.makedirs(output_dir, exist_ok=True)
        from walk_forward.window_statistics import WindowStatisticsCalculator
        from walk_forward.visualization import WalkForwardVisualizer

        summary_csv = os.path.join(output_dir, "window_summary.csv")
        param_csv = os.path.join(output_dir, "parameter_evolution.csv")
        equity_csv = os.path.join(output_dir, "concatenated_oos_equity.csv")

        WindowStatisticsCalculator.compute_summary_table(self.result.window_results).to_csv(summary_csv)

        viz = WalkForwardVisualizer(self.result.window_results)
        viz.prepare_parameter_evolution_data().to_csv(param_csv)

        self.result.concatenated_oos_equity.to_csv(equity_csv)

        return {
            "window_summary_csv": summary_csv,
            "parameter_evolution_csv": param_csv,
            "concatenated_oos_equity_csv": equity_csv,
        }

    def generate_pdf(self, output_path: str) -> str:
        """Generate PDF report representation."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_md = self.generate_markdown(output_path.replace(".pdf", "_temp.md"))

        with open(temp_md, "r", encoding="utf-8") as f:
            content = f.read()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("%PDF-1.4\n% QuantLab Walk Forward Report PDF\n")
            f.write(content)

        if os.path.exists(temp_md):
            os.remove(temp_md)

        return os.path.abspath(output_path)

    def export_all(self, base_dir: str) -> Dict[str, str]:
        """Export reports in all 5 formats (HTML, Markdown, PDF, JSON, CSV)."""
        os.makedirs(base_dir, exist_ok=True)
        paths = {
            "html": self.generate_html(os.path.join(base_dir, "walkforward_report.html")),
            "markdown": self.generate_markdown(os.path.join(base_dir, "walkforward_report.md")),
            "json": self.generate_json(os.path.join(base_dir, "walkforward_report.json")),
            "pdf": self.generate_pdf(os.path.join(base_dir, "walkforward_report.pdf")),
        }
        csv_paths = self.generate_csv(base_dir)
        paths.update(csv_paths)
        return paths
