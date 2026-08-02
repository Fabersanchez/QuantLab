"""
QuantLab Reinforcement Learning - Multi-Format Report Generator.

Exports RL research results in HTML Dashboard, Markdown, PDF, JSON, and CSV formats.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from reinforcement_learning.visualization import RLVisualizer


class RLReportGenerator:
    """Institutional RL Research Report Exporter (HTML, Markdown, PDF, JSON, CSV)."""

    def __init__(self, result: Any) -> None:
        """Initialize RLReportGenerator with an RLEngineResult object."""
        self.result = result

    def generate_html(self, output_path: str) -> str:
        """Generate standalone HTML dashboard with SVG reward and action charts."""
        m = self.result.metrics
        viz = RLVisualizer()

        reward_svg = viz.generate_reward_curve_svg(self.result.episode_rewards, width=850)
        action_svg = viz.generate_action_distribution_svg(self.result.action_distribution, width=850)
        loss_svg = viz.generate_learning_progress_svg(self.result.loss_history, width=850)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantLab RL Report - {self.result.algorithm}</title>
    <style>
        :root {{
            --bg: #0f172a; --card: #1e293b; --text: #f8fafc;
            --muted: #94a3b8; --green: #10b981; --blue: #3b82f6;
            --red: #ef4444; --border: #334155;
        }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }}
        h1 {{ color: var(--blue); margin: 0 0 6px 0; font-size: 24px; }}
        .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 24px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 28px; }}
        .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
        .card-title {{ font-size: 11px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }}
        .card-value {{ font-size: 22px; font-weight: 700; }}
        .pos {{ color: var(--green); }} .neg {{ color: var(--red); }}
        .section {{ font-size: 18px; font-weight: 600; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin: 28px 0 14px 0; }}
        .chart {{ margin-bottom: 24px; }}
    </style>
</head>
<body>
    <h1>QuantLab Reinforcement Learning Research Report</h1>
    <div class="sub">Algorithm: <strong>{self.result.algorithm}</strong> &nbsp;|&nbsp; Agent ID: <code>{self.result.agent_id}</code></div>

    <div class="grid">
        <div class="card"><div class="card-title">Mean Episode Reward</div><div class="card-value {'pos' if m.get('mean_reward',0)>=0 else 'neg'}">{m.get('mean_reward',0):.3f}</div></div>
        <div class="card"><div class="card-title">Best Episode Reward</div><div class="card-value pos">{m.get('best_reward',0):.3f}</div></div>
        <div class="card"><div class="card-title">Sharpe Ratio</div><div class="card-value {'pos' if m.get('sharpe_ratio',0)>=0 else 'neg'}">{m.get('sharpe_ratio',0):.3f}</div></div>
        <div class="card"><div class="card-title">Sortino Ratio</div><div class="card-value {'pos' if m.get('sortino_ratio',0)>=0 else 'neg'}">{m.get('sortino_ratio',0):.3f}</div></div>
        <div class="card"><div class="card-title">Max Drawdown</div><div class="card-value neg">{m.get('max_drawdown_pct',0)*100:.1f}%</div></div>
        <div class="card"><div class="card-title">Win Rate</div><div class="card-value">{m.get('win_rate',0)*100:.1f}%</div></div>
        <div class="card"><div class="card-title">Profit Factor</div><div class="card-value pos">{m.get('profit_factor',0):.2f}</div></div>
        <div class="card"><div class="card-title">Total Episodes</div><div class="card-value">{self.result.n_episodes}</div></div>
    </div>

    <div class="section">Episode Reward Curve</div>
    <div class="chart">{reward_svg}</div>

    <div class="section">Agent Action Distribution</div>
    <div class="chart">{action_svg}</div>

    <div class="section">Policy Training Loss</div>
    <div class="chart">{loss_svg}</div>
</body>
</html>"""

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return os.path.abspath(output_path)

    def generate_markdown(self, output_path: str) -> str:
        """Generate Markdown report file."""
        m = self.result.metrics
        md = f"""# QuantLab Reinforcement Learning Report

**Algorithm**: `{self.result.algorithm}`  
**Agent ID**: `{self.result.agent_id}`  
**Episodes Trained**: {self.result.n_episodes}

---

## Performance Metrics

| Metric | Value |
| :--- | ---: |
| **Mean Episode Reward** | **{m.get('mean_reward', 0):.4f}** |
| **Best Episode Reward** | **{m.get('best_reward', 0):.4f}** |
| **Worst Episode Reward** | **{m.get('worst_reward', 0):.4f}** |
| **Sharpe Ratio** | **{m.get('sharpe_ratio', 0):.4f}** |
| **Sortino Ratio** | **{m.get('sortino_ratio', 0):.4f}** |
| **Max Drawdown** | **{m.get('max_drawdown_pct', 0)*100:.2f}%** |
| **Win Rate** | **{m.get('win_rate', 0)*100:.2f}%** |
| **Profit Factor** | **{m.get('profit_factor', 0):.4f}** |
| **Stability** | **{m.get('stability', 0):.4f}** |
"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        return os.path.abspath(output_path)

    def generate_json(self, output_path: str) -> str:
        """Export JSON dump of RL results."""
        data = {
            "algorithm": self.result.algorithm,
            "agent_id": self.result.agent_id,
            "n_episodes": self.result.n_episodes,
            "metrics": self.result.metrics,
            "episode_rewards": self.result.episode_rewards,
            "action_distribution": {str(k): v for k, v in self.result.action_distribution.items()},
            "execution_time_seconds": self.result.execution_time_seconds,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return os.path.abspath(output_path)

    def generate_csv(self, output_dir: str) -> Dict[str, str]:
        """Export CSV files for episode rewards and action distribution."""
        os.makedirs(output_dir, exist_ok=True)

        rewards_csv = os.path.join(output_dir, "episode_rewards.csv")
        pd.DataFrame({"episode": range(len(self.result.episode_rewards)), "reward": self.result.episode_rewards}).to_csv(rewards_csv, index=False)

        actions_csv = os.path.join(output_dir, "action_distribution.csv")
        pd.DataFrame([{"action": k, "count": v} for k, v in self.result.action_distribution.items()]).to_csv(actions_csv, index=False)

        return {"episode_rewards_csv": rewards_csv, "action_distribution_csv": actions_csv}

    def generate_pdf(self, output_path: str) -> str:
        """Generate PDF-placeholder report from Markdown."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        md_path = output_path.replace(".pdf", "_tmp.md")
        self.generate_markdown(md_path)
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("%PDF-1.4\n% QuantLab RL Report\n" + content)
        if os.path.exists(md_path):
            os.remove(md_path)
        return os.path.abspath(output_path)

    def export_all(self, base_dir: str) -> Dict[str, str]:
        """Export all 5 report formats to the given base directory."""
        os.makedirs(base_dir, exist_ok=True)
        paths = {
            "html": self.generate_html(os.path.join(base_dir, "rl_report.html")),
            "markdown": self.generate_markdown(os.path.join(base_dir, "rl_report.md")),
            "json": self.generate_json(os.path.join(base_dir, "rl_report.json")),
            "pdf": self.generate_pdf(os.path.join(base_dir, "rl_report.pdf")),
        }
        paths.update(self.generate_csv(base_dir))
        return paths
