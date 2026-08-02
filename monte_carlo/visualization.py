"""
QuantLab Monte Carlo Visualizer & Fan Chart Generator.

Generates chart data and SVG graphics for Equity Fan Charts, Drawdown Fan Charts,
Return Distribution Histograms, Probability Curves, and Risk Heatmaps.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from monte_carlo.simulation_runner import SimulationIterationResult


class MonteCarloVisualizer:
    """Institutional Visual Analytics Generator for Monte Carlo Simulations."""

    def __init__(self, simulation_results: List[SimulationIterationResult]) -> None:
        """Initialize MonteCarloVisualizer."""
        self.results = simulation_results

    def compute_equity_fan_bands(
        self, percentiles: List[float] = [5.0, 25.0, 50.0, 75.0, 95.0]
    ) -> pd.DataFrame:
        """Compute percentile equity curves across all simulation iterations at each step.

        Returns:
            pd.DataFrame containing percentile columns ('p5', 'p25', 'p50', 'p75', 'p95').
        """
        if not self.results:
            return pd.DataFrame()

        # Find max step length
        max_len = max(len(it.equity_series) for it in self.results)
        matrix = np.full((len(self.results), max_len), np.nan)

        for i, it in enumerate(self.results):
            s = it.equity_series
            matrix[i, : len(s)] = s
            # Forward fill if shorter
            if len(s) < max_len and len(s) > 0:
                matrix[i, len(s) :] = s[-1]

        data = {}
        for p in percentiles:
            col_name = f"p{int(p)}"
            data[col_name] = np.nanpercentile(matrix, p, axis=0)

        return pd.DataFrame(data)

    def generate_equity_fan_chart_svg(self, width: int = 850, height: int = 350) -> str:
        """Generate SVG Equity Fan Chart with shaded percentile bands (5th-95th, 25th-75th, 50th median)."""
        df_fan = self.compute_equity_fan_bands()
        if df_fan.empty:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Monte Carlo Data</text></svg>'

        margin_left = 60
        margin_right = 20
        margin_top = 20
        margin_bottom = 40
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        min_val = float(df_fan["p5"].min())
        max_val = float(df_fan["p95"].max())
        rng = max(1.0, max_val - min_val)

        n = len(df_fan)

        def _get_pts(col_name: str) -> List[Tuple[float, float]]:
            pts = []
            for idx, val in enumerate(df_fan[col_name]):
                x = margin_left + (idx / max(1, n - 1)) * plot_w
                y = margin_top + (1.0 - (val - min_val) / rng) * plot_h
                pts.append((x, y))
            return pts

        pts_p5 = _get_pts("p5")
        pts_p25 = _get_pts("p25")
        pts_p50 = _get_pts("p50")
        pts_p75 = _get_pts("p75")
        pts_p95 = _get_pts("p95")

        # Outer band (5th to 95th) polygon
        outer_poly = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts_p95]) + " " + " ".join([f"{x:.1f},{y:.1f}" for x, y in reversed(pts_p5)])

        # Inner band (25th to 75th) polygon
        inner_poly = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts_p75]) + " " + " ".join([f"{x:.1f},{y:.1f}" for x, y in reversed(pts_p25)])

        # Median line
        med_pts = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts_p50])

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <!-- Axes & Labels -->
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <text x="10" y="{margin_top + 10}" fill="#94a3b8" font-size="11">${max_val:,.0f}</text>
        <text x="10" y="{height - margin_bottom - 5}" fill="#94a3b8" font-size="11">${min_val:,.0f}</text>

        <!-- Outer Band 5th - 95th -->
        <polygon points="{outer_poly}" fill="#3b82f6" fill-opacity="0.15" stroke="none" />

        <!-- Inner Band 25th - 75th -->
        <polygon points="{inner_poly}" fill="#3b82f6" fill-opacity="0.30" stroke="none" />

        <!-- Median 50th Line -->
        <polyline points="{med_pts}" fill="none" stroke="#10b981" stroke-width="2.5" />

        <!-- Legend -->
        <text x="{width - 220}" y="{margin_top + 15}" fill="#10b981" font-size="11" font-weight="bold">Median (50th)</text>
        <text x="{width - 220}" y="{margin_top + 30}" fill="#60a5fa" font-size="11">25th - 75th Band</text>
        <text x="{width - 220}" y="{margin_top + 45}" fill="#93c5fd" font-size="11">5th - 95th Outer Band</text>
    </svg>"""
        return svg

    def generate_histogram_svg(self, width: int = 850, height: int = 250, n_bins: int = 20) -> str:
        """Generate SVG histogram of Monte Carlo net profit outcomes."""
        if not self.results:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Data</text></svg>'

        profits = [it.net_profit for it in self.results]
        counts, bin_edges = np.histogram(profits, bins=n_bins)

        margin_left = 50
        margin_right = 20
        margin_top = 20
        margin_bottom = 35
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        max_count = max(1, max(counts))
        bar_w = plot_w / n_bins

        bars_svg = []
        for i in range(n_bins):
            c = counts[i]
            h = (c / max_count) * plot_h
            x = margin_left + i * bar_w
            y = height - margin_bottom - h
            center_val = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            color = "#10b981" if center_val >= 0 else "#ef4444"

            bars_svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{h:.1f}" fill="{color}" opacity="0.85"/>')

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <text x="20" y="20" fill="#f8fafc" font-size="12" font-weight="bold">Net Profit Outcome Distribution</text>
        {''.join(bars_svg)}
    </svg>"""
        return svg
