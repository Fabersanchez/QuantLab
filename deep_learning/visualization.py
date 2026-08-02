"""
QuantLab Deep Learning Visual Analytics & SVG Chart Generator.

Generates inline SVG graphics for Training/Validation Loss Curves,
Learning Rate Schedule, Confusion Matrix Heatmaps, and Feature/Attention Map Heatmaps.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class DLVisualizer:
    """Institutional Visual Analytics Generator for Deep Learning."""

    @staticmethod
    def generate_loss_curves_svg(
        loss_history: Dict[str, List[float]], width: int = 600, height: int = 300
    ) -> str:
        """Generate inline SVG string of Epoch Training vs Validation Loss curves."""
        train_loss = loss_history.get("train_loss", [])
        val_loss = loss_history.get("val_loss", [])

        if not train_loss:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Loss History</text></svg>'

        n_epochs = len(train_loss)
        max_val = max(1e-4, max(max(train_loss), max(val_loss) if val_loss else 0.0))
        min_val = min(min(train_loss), min(val_loss) if val_loss else 0.0)
        rng = max(1e-4, max_val - min_val)

        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 40
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        def _get_points(loss_list: List[float]) -> str:
            pts = []
            for i, l in enumerate(loss_list):
                x = margin_left + (i / max(1, n_epochs - 1)) * plot_w
                y = margin_top + (1.0 - (l - min_val) / rng) * plot_h
                pts.append(f"{x:.1f},{y:.1f}")
            return " ".join(pts)

        t_pts = _get_points(train_loss)
        v_pts = _get_points(val_loss) if val_loss else ""

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <text x="20" y="20" fill="#f8fafc" font-size="13" font-weight="bold">Epoch Training &amp; Validation Loss Curves</text>
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>

        <!-- Train Loss Curve (Blue) -->
        <polyline points="{t_pts}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>

        <!-- Val Loss Curve (Green) -->
        <polyline points="{v_pts}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="4"/>

        <!-- Legend -->
        <text x="{width - 160}" y="{margin_top + 15}" fill="#3b82f6" font-size="11" font-weight="bold">Train Loss</text>
        <text x="{width - 160}" y="{margin_top + 30}" fill="#10b981" font-size="11" font-weight="bold">Val Loss</text>
    </svg>"""
        return svg

    @staticmethod
    def generate_attention_map_svg(
        attention_weights: np.ndarray, width: int = 500, height: int = 250
    ) -> str:
        """Generate inline SVG heatmap of temporal attention weights over sequence steps."""
        if attention_weights.ndim != 2:
            attention_weights = np.atleast_2d(attention_weights)

        n_rows, n_cols = attention_weights.shape
        margin_left = 40
        margin_top = 30
        plot_w = width - margin_left - 20
        plot_h = height - margin_top - 30

        cell_w = plot_w / n_cols
        cell_h = plot_h / n_rows

        max_w = max(1e-4, float(np.max(attention_weights)))

        rects = []
        for i in range(n_rows):
            for j in range(n_cols):
                val = float(attention_weights[i, j])
                opacity = 0.1 + 0.9 * (val / max_w)
                x = margin_left + j * cell_w
                y = margin_top + i * cell_h
                rects.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-1:.1f}" height="{cell_h-1:.1f}" fill="#8b5cf6" opacity="{opacity:.2f}"/>'
                )

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <text x="20" y="20" fill="#f8fafc" font-size="13" font-weight="bold">Temporal Attention Heatmap</text>
        {''.join(rects)}
    </svg>"""
        return svg
