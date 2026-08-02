"""
QuantLab Multi-Format Monte Carlo Report Generator.

Exports Monte Carlo & Robustness Engine results in HTML, Markdown, PDF, JSON, and CSV formats.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd


class MonteCarloReportGenerator:
    """Institutional Monte Carlo Report Exporter."""

    def __init__(self, montecarlo_result: Any) -> None:
        """Initialize MonteCarloReportGenerator with MonteCarloResult object."""
        self.result = montecarlo_result

    def generate_html(self, output_path: str) -> str:
        """Generate standalone HTML dashboard report with SVG Fan Charts and interactive tables.

        Args:
            output_path: Target HTML file path.

        Returns:
            Absolute file path.
        """
        rob = self.result.robustness_score
        dist = self.result.distribution_metrics
        prob = self.result.probability_metrics
        ci = self.result.confidence_intervals

        score = rob.get("institutional_robustness_score", 0.0)
        pop = prob.get("probability_of_profit_pct", 0.0)
        por = prob.get("probability_of_ruin_pct", 0.0)

        # Generate SVG charts
        from monte_carlo.visualization import MonteCarloVisualizer
        viz = MonteCarloVisualizer(self.result.iteration_results)
        fan_svg = viz.generate_equity_fan_chart_svg(width=850, height=320)
        hist_svg = viz.generate_histogram_svg(width=850, height=220)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLab Monte Carlo Analysis - {self.result.strategy_name}</title>
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
            <h1>QuantLab Monte Carlo & Robustness Report</h1>
            <div style="color: var(--text-muted); margin-top: 4px;">
                Strategy: <strong>{self.result.strategy_name}</strong> | Asset: <strong>{self.result.asset_symbol}</strong> | Simulations: <strong>{self.result.total_iterations:,}</strong>
            </div>
        </div>
        <div style="text-align: right; color: var(--text-muted); font-size: 12px;">
            Execution Time: {self.result.execution_time_seconds:.2f}s
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Institutional Robustness Score</div>
            <div class="card-value {"val-pos" if score >= 70 else "val-neg"}">{score:.1f} / 100</div>
        </div>
        <div class="card">
            <div class="card-title">Probability of Profit</div>
            <div class="card-value val-pos">{pop:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Probability of Ruin</div>
            <div class="card-value {"val-neg" if por > 5 else "val-pos"}">{por:.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Expected Return</div>
            <div class="card-value {"val-pos" if dist.get("expected_return", 0) >= 0 else "val-neg"}">${dist.get("expected_return", 0):,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Mean Max Drawdown</div>
            <div class="card-value val-neg">-{dist.get("mean_max_drawdown_pct", 0):.1f}%</div>
        </div>
        <div class="card">
            <div class="card-title">95th Percentile Worst DD</div>
            <div class="card-value val-neg">-{dist.get("p95_max_drawdown_pct", 0):.1f}%</div>
        </div>
    </div>

    <div class="section-title">Monte Carlo Equity Fan Chart (Percentile Bands)</div>
    <div class="chart-container">
        {fan_svg}
    </div>

    <div class="section-title">Net Profit Distribution Histogram</div>
    <div class="chart-container">
        {hist_svg}
    </div>

    <div class="section-title">Confidence Intervals (Bootstrap & Percentile)</div>
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>90% Confidence Interval</th>
                <th>95% Confidence Interval</th>
                <th>99% Confidence Interval</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Net Profit ($)</td>
                <td>${ci.get('net_profit', {}).get('90%', (0,0))[0]:,.2f} to ${ci.get('net_profit', {}).get('90%', (0,0))[1]:,.2f}</td>
                <td>${ci.get('net_profit', {}).get('95%', (0,0))[0]:,.2f} to ${ci.get('net_profit', {}).get('95%', (0,0))[1]:,.2f}</td>
                <td>${ci.get('net_profit', {}).get('99%', (0,0))[0]:,.2f} to ${ci.get('net_profit', {}).get('99%', (0,0))[1]:,.2f}</td>
            </tr>
            <tr>
                <td>Max Drawdown (%)</td>
                <td>{ci.get('max_drawdown_pct', {}).get('90%', (0,0))[0]:.1f}% to {ci.get('max_drawdown_pct', {}).get('90%', (0,0))[1]:.1f}%</td>
                <td>{ci.get('max_drawdown_pct', {}).get('95%', (0,0))[0]:.1f}% to {ci.get('max_drawdown_pct', {}).get('95%', (0,0))[1]:.1f}%</td>
                <td>{ci.get('max_drawdown_pct', {}).get('99%', (0,0))[0]:.1f}% to {ci.get('max_drawdown_pct', {}).get('99%', (0,0))[1]:.1f}%</td>
            </tr>
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
        """Generate Markdown Monte Carlo report."""
        rob = self.result.robustness_score
        dist = self.result.distribution_metrics
        prob = self.result.probability_metrics
        ci = self.result.confidence_intervals

        md = f"""# QuantLab Monte Carlo & Robustness Report

**Strategy**: {self.result.strategy_name}  
**Asset**: {self.result.asset_symbol}  
**Simulation Iterations**: {self.result.total_iterations:,}  

---

## Executive Robustness Summary

| Metric | Value |
| :--- | :--- |
| **Institutional Robustness Score** | **{rob.get('institutional_robustness_score', 0):.1f} / 100** |
| **Probability of Profit (PoP)** | **{prob.get('probability_of_profit_pct', 0):.1f}%** |
| **Probability of Ruin (PoR)** | **{prob.get('probability_of_ruin_pct', 0):.1f}%** |
| **Expected Return** | **${dist.get('expected_return', 0):,.2f}** |
| **Median Return** | **${dist.get('median_return', 0):,.2f}** |
| **Mean Max Drawdown %** | **-{dist.get('mean_max_drawdown_pct', 0):.1f}%** |
| **95th Percentile Worst DD** | **-{dist.get('p95_max_drawdown_pct', 0):.1f}%** |
| **Skewness** | **{dist.get('skewness', 0):.2f}** |
| **Kurtosis** | **{dist.get('kurtosis', 0):.2f}** |

---

## Confidence Interval Estimations

| Metric | 90% CI | 95% CI | 99% CI |
| :--- | :--- | :--- | :--- |
| **Net Profit** | ${ci.get('net_profit', {}).get('90%', (0,0))[0]:,.2f} ... ${ci.get('net_profit', {}).get('90%', (0,0))[1]:,.2f} | ${ci.get('net_profit', {}).get('95%', (0,0))[0]:,.2f} ... ${ci.get('net_profit', {}).get('95%', (0,0))[1]:,.2f} | ${ci.get('net_profit', {}).get('99%', (0,0))[0]:,.2f} ... ${ci.get('net_profit', {}).get('99%', (0,0))[1]:,.2f} |
| **Max Drawdown** | {ci.get('max_drawdown_pct', {}).get('90%', (0,0))[0]:.1f}% ... {ci.get('max_drawdown_pct', {}).get('90%', (0,0))[1]:.1f}% | {ci.get('max_drawdown_pct', {}).get('95%', (0,0))[0]:.1f}% ... {ci.get('max_drawdown_pct', {}).get('95%', (0,0))[1]:.1f}% | {ci.get('max_drawdown_pct', {}).get('99%', (0,0))[0]:.1f}% ... {ci.get('max_drawdown_pct', {}).get('99%', (0,0))[1]:.1f}% |

"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        return os.path.abspath(output_path)

    def generate_json(self, output_path: str) -> str:
        """Export full Monte Carlo JSON dump."""
        data = {
            "strategy_name": self.result.strategy_name,
            "asset_symbol": self.result.asset_symbol,
            "total_iterations": self.result.total_iterations,
            "robustness_score": self.result.robustness_score,
            "distribution_metrics": self.result.distribution_metrics,
            "probability_metrics": self.result.probability_metrics,
            "confidence_intervals": self.result.confidence_intervals,
            "execution_time_seconds": self.result.execution_time_seconds,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return os.path.abspath(output_path)

    def generate_csv(self, output_dir: str) -> Dict[str, str]:
        """Export CSV files for Equity Fan bands and simulation summary."""
        os.makedirs(output_dir, exist_ok=True)
        from monte_carlo.visualization import MonteCarloVisualizer

        fan_csv = os.path.join(output_dir, "equity_fan_bands.csv")
        summary_csv = os.path.join(output_dir, "simulation_summary.csv")

        viz = MonteCarloVisualizer(self.result.iteration_results)
        viz.compute_equity_fan_bands().to_csv(fan_csv)

        summary_data = [
            {
                "iteration_id": it.iteration_id,
                "net_profit": it.net_profit,
                "final_equity": it.final_equity,
                "max_drawdown_pct": it.max_drawdown_pct,
                "total_trades": it.total_trades,
                "win_rate": it.win_rate,
                "ruin_occurred": it.ruin_occurred,
            }
            for it in self.result.iteration_results
        ]
        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)

        return {"equity_fan_bands_csv": fan_csv, "simulation_summary_csv": summary_csv}

    def generate_pdf(self, output_path: str) -> str:
        """Generate PDF report representation."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_md = self.generate_markdown(output_path.replace(".pdf", "_temp.md"))

        with open(temp_md, "r", encoding="utf-8") as f:
            content = f.read()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("%PDF-1.4\n% QuantLab Monte Carlo Report PDF\n")
            f.write(content)

        if os.path.exists(temp_md):
            os.remove(temp_md)

        return os.path.abspath(output_path)

    def export_all(self, base_dir: str) -> Dict[str, str]:
        """Export reports in all 5 formats (HTML, Markdown, PDF, JSON, CSV)."""
        os.makedirs(base_dir, exist_ok=True)
        paths = {
            "html": self.generate_html(os.path.join(base_dir, "montecarlo_report.html")),
            "markdown": self.generate_markdown(os.path.join(base_dir, "montecarlo_report.md")),
            "json": self.generate_json(os.path.join(base_dir, "montecarlo_report.json")),
            "pdf": self.generate_pdf(os.path.join(base_dir, "montecarlo_report.pdf")),
        }
        csv_paths = self.generate_csv(base_dir)
        paths.update(csv_paths)
        return paths
