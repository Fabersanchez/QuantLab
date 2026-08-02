"""
QuantLab Walk Forward Visualizer & Chart Data Generator.

Generates chart data structures and inline SVG visualizations for Concatenated OOS Equity Curves,
Window Comparison Bar Charts, Parameter Evolution timelines, Heatmaps, and Drawdown Timelines.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from walk_forward.validation_runner import ValidationStepResult


class WalkForwardVisualizer:
    """Institutional Visual Analytics Generator for Walk Forward Analysis."""

    def __init__(self, step_results: List[ValidationStepResult]) -> None:
        """Initialize WalkForwardVisualizer."""
        self.step_results = step_results

    def generate_equity_comparison_svg(
        self, concatenated_oos_equity: pd.DataFrame, width: int = 800, height: int = 350
    ) -> str:
        """Generate inline SVG string of concatenated OOS equity curve vs initial capital baseline."""
        if concatenated_oos_equity.empty or "equity" not in concatenated_oos_equity.columns:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Equity Data</text></svg>'

        eq_series = concatenated_oos_equity["equity"].values
        min_v = float(min(eq_series))
        max_v = float(max(eq_series))
        rng = max(1.0, max_v - min_v)

        margin_left = 60
        margin_right = 20
        margin_top = 20
        margin_bottom = 40
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        points = []
        n = len(eq_series)
        for idx, val in enumerate(eq_series):
            x = margin_left + (idx / max(1, n - 1)) * plot_w
            y = margin_top + (1.0 - (val - min_v) / rng) * plot_h
            points.append(f"{x:.1f},{y:.1f}")

        polyline_pts = " ".join(points)
        start_cap = float(eq_series[0])
        baseline_y = margin_top + (1.0 - (start_cap - min_v) / rng) * plot_h

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <!-- Grid & Baseline -->
        <line x1="{margin_left}" y1="{baseline_y:.1f}" x2="{width - margin_right}" y2="{baseline_y:.1f}" stroke="#475569" stroke-dasharray="4" stroke-width="1"/>
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>

        <!-- Y Axis Labels -->
        <text x="10" y="{margin_top + 10}" fill="#94a3b8" font-size="11">${max_v:,.0f}</text>
        <text x="10" y="{height - margin_bottom - 5}" fill="#94a3b8" font-size="11">${min_v:,.0f}</text>

        <!-- OOS Equity Polyline -->
        <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="{polyline_pts}" />
    </svg>"""
        return svg

    def generate_window_comparison_svg(self, width: int = 800, height: int = 300) -> str:
        """Generate inline SVG grouped bar chart comparing IS Sharpe vs OOS Sharpe per window."""
        if not self.step_results:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Window Data</text></svg>'

        n_win = len(self.step_results)
        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 40
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        is_sharpes = [float(s.is_metrics.get("sharpe_ratio", 0.0)) for s in self.step_results]
        oos_sharpes = [float(s.oos_metrics.get("sharpe_ratio", 0.0)) for s in self.step_results]
        max_s = max(1.0, max(max(is_sharpes), max(oos_sharpes)))

        bar_group_w = plot_w / max(1, n_win)
        bar_w = max(4, (bar_group_w - 12) / 2)

        bars_svg = []
        for i in range(n_win):
            is_s = is_sharpes[i]
            oos_s = oos_sharpes[i]

            is_h = (max(0.0, is_s) / max_s) * plot_h
            oos_h = (max(0.0, oos_s) / max_s) * plot_h

            x_group = margin_left + i * bar_group_w + 6
            x_is = x_group
            x_oos = x_group + bar_w + 2

            y_is = height - margin_bottom - is_h
            y_oos = height - margin_bottom - oos_h

            bars_svg.append(f'<rect x="{x_is:.1f}" y="{y_is:.1f}" width="{bar_w:.1f}" height="{is_h:.1f}" fill="#3b82f6" rx="2"/>')
            bars_svg.append(f'<rect x="{x_oos:.1f}" y="{y_oos:.1f}" width="{bar_w:.1f}" height="{oos_h:.1f}" fill="#10b981" rx="2"/>')
            bars_svg.append(f'<text x="{x_group + bar_w:.1f}" y="{height - 15}" fill="#94a3b8" font-size="10" text-anchor="middle">W{i}</text>')

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <!-- Title & Legend -->
        <text x="20" y="20" fill="#f8fafc" font-size="12" font-weight="bold">Sharpe Ratio: In-Sample vs Out-of-Sample</text>
        <rect x="{width - 160}" y="10" width="12" height="12" fill="#3b82f6" rx="2"/>
        <text x="{width - 142}" y="20" fill="#94a3b8" font-size="11">In-Sample</text>
        <rect x="{width - 80}" y="10" width="12" height="12" fill="#10b981" rx="2"/>
        <text x="{width - 62}" y="20" fill="#94a3b8" font-size="11">OOS</text>

        <!-- Baseline -->
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>

        <!-- Bars -->
        {''.join(bars_svg)}
    </svg>"""
        return svg

    def prepare_parameter_evolution_data(self) -> pd.DataFrame:
        """Export DataFrame tracking parameter changes window by window."""
        if not self.step_results:
            return pd.DataFrame()

        rows = []
        for s in self.step_results:
            row = {"window": s.window_index}
            row.update(s.best_params)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.set_index("window", inplace=True)
        return df
