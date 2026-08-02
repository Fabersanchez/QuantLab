"""
QuantLab Reinforcement Learning - Visual Analytics & SVG Chart Generator.

Generates inline SVG graphics for:
  - Episode Reward Curves
  - Training Loss Curves
  - Action Distribution Bar Charts
  - Policy Evolution Heatmaps
  - Learning Progress Charts
"""

from typing import Dict, List, Optional
import numpy as np


class RLVisualizer:
    """Institutional Visual Analytics Generator for Reinforcement Learning."""

    @staticmethod
    def generate_reward_curve_svg(
        episode_rewards: List[float],
        width: int = 700,
        height: int = 280,
        title: str = "Episode Reward Curve",
    ) -> str:
        """Generate inline SVG of episode reward progression.

        Args:
            episode_rewards: List of total rewards per training episode.
            width: SVG canvas width in pixels.
            height: SVG canvas height in pixels.
            title: Chart title.

        Returns:
            Inline SVG string.
        """
        if not episode_rewards:
            return f'<svg width="{width}" height="{height}"><text x="20" y="40" fill="#94a3b8">No reward data</text></svg>'

        n = len(episode_rewards)
        arr = np.array(episode_rewards, dtype=np.float64)
        rolling = np.convolve(arr, np.ones(min(10, n)) / min(10, n), mode="same")

        mx = max(1e-4, float(arr.max()))
        mn = min(-1e-4, float(arr.min()))
        rng = max(1e-4, mx - mn)

        ml, mr, mt, mb = 55, 20, 35, 40
        pw = width - ml - mr
        ph = height - mt - mb

        def _pts(values):
            pts = []
            for i, v in enumerate(values):
                x = ml + (i / max(1, n - 1)) * pw
                y = mt + (1.0 - (v - mn) / rng) * ph
                pts.append(f"{x:.1f},{y:.1f}")
            return " ".join(pts)

        # Y axis labels
        y_labels = ""
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = mn + frac * rng
            y_px = mt + (1.0 - frac) * ph
            y_labels += f'<text x="{ml-5}" y="{y_px:.1f}" fill="#64748b" font-size="10" text-anchor="end">{val:.2f}</text>'
            y_labels += f'<line x1="{ml}" y1="{y_px:.1f}" x2="{width-mr}" y2="{y_px:.1f}" stroke="#334155" stroke-width="0.5" stroke-dasharray="3"/>'

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
            style="background:#0f172a; border-radius:8px; font-family:sans-serif;">
            <text x="20" y="22" fill="#f8fafc" font-size="13" font-weight="bold">{title}</text>
            {y_labels}
            <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            <line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            <polyline points="{_pts(arr)}" fill="none" stroke="#3b82f6" stroke-width="1.5" opacity="0.5"/>
            <polyline points="{_pts(rolling)}" fill="none" stroke="#10b981" stroke-width="2.5"/>
            <text x="{width-160}" y="{mt+15}" fill="#3b82f6" font-size="11">Raw Reward</text>
            <text x="{width-160}" y="{mt+30}" fill="#10b981" font-size="11">Rolling Avg (10)</text>
        </svg>"""
        return svg

    @staticmethod
    def generate_action_distribution_svg(
        action_counts: Dict[int, int],
        action_labels: Optional[Dict[int, str]] = None,
        width: int = 600,
        height: int = 240,
    ) -> str:
        """Generate inline SVG bar chart of agent action distribution.

        Args:
            action_counts: Dict mapping action index to occurrence count.
            action_labels: Optional dict mapping action index to label string.
            width: SVG canvas width.
            height: SVG canvas height.

        Returns:
            Inline SVG string.
        """
        if not action_counts:
            return f'<svg width="{width}" height="{height}"><text x="20" y="40" fill="#94a3b8">No action data</text></svg>'

        labels = action_labels or {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE", 4: "PARTIAL", 5: "MOD_SL", 6: "MOD_TP"}
        keys = sorted(action_counts.keys())
        values = np.array([action_counts.get(k, 0) for k in keys], dtype=np.float64)
        total = max(1.0, values.sum())

        ml, mr, mt, mb = 40, 20, 35, 40
        pw = width - ml - mr
        ph = height - mt - mb
        bar_w = pw / len(keys) * 0.7
        gap = pw / len(keys)

        COLORS = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]
        bars = ""
        for i, k in enumerate(keys):
            bar_h = float((values[i] / total) * ph)
            x = ml + i * gap + (gap - bar_w) / 2
            y = mt + ph - bar_h
            color = COLORS[i % len(COLORS)]
            lbl = labels.get(k, str(k))
            pct = values[i] / total * 100
            bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>'
            bars += f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" fill="#f8fafc" font-size="10" text-anchor="middle">{pct:.1f}%</text>'
            bars += f'<text x="{x+bar_w/2:.1f}" y="{height-15:.1f}" fill="#94a3b8" font-size="10" text-anchor="middle">{lbl}</text>'

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
            style="background:#0f172a; border-radius:8px; font-family:sans-serif;">
            <text x="20" y="22" fill="#f8fafc" font-size="13" font-weight="bold">Agent Action Distribution</text>
            <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            <line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            {bars}
        </svg>"""
        return svg

    @staticmethod
    def generate_learning_progress_svg(
        loss_history: List[float],
        width: int = 700,
        height: int = 220,
    ) -> str:
        """Generate inline SVG of policy training loss progression."""
        if not loss_history:
            return f'<svg width="{width}" height="{height}"><text x="20" y="40" fill="#94a3b8">No loss data</text></svg>'

        arr = np.array(loss_history, dtype=np.float64)
        n = len(arr)
        mx = max(1e-4, float(arr.max()))
        mn = 0.0
        rng = max(1e-4, mx - mn)

        ml, mr, mt, mb = 55, 20, 35, 35
        pw = width - ml - mr
        ph = height - mt - mb

        pts = []
        for i, v in enumerate(arr):
            x = ml + (i / max(1, n - 1)) * pw
            y = mt + (1.0 - (v - mn) / rng) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        pts_str = " ".join(pts)

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"
            style="background:#0f172a; border-radius:8px; font-family:sans-serif;">
            <text x="20" y="22" fill="#f8fafc" font-size="13" font-weight="bold">Policy Training Loss</text>
            <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            <line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#475569" stroke-width="1"/>
            <polyline points="{pts_str}" fill="none" stroke="#f59e0b" stroke-width="2"/>
        </svg>"""
        return svg
