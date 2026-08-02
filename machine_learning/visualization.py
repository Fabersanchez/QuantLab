"""
QuantLab Machine Learning Visual Analytics & SVG Chart Generator.

Generates SVG chart graphics for Confusion Matrices, ROC Curves,
PR Curves, Feature Importance Bar Charts, and Calibration Curves.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class MLVisualizer:
    """Institutional Visual Analytics Generator for Machine Learning."""

    @staticmethod
    def generate_confusion_matrix_svg(
        cm: np.ndarray, width: int = 400, height: int = 300, class_names: Optional[List[str]] = None
    ) -> str:
        """Generate inline SVG string of confusion matrix heatmap."""
        if cm.ndim != 2:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">Invalid CM</text></svg>'

        n_rows, n_cols = cm.shape
        labels = class_names or [f"Class {i}" for i in range(n_rows)]

        margin_left = 60
        margin_top = 40
        plot_w = width - margin_left - 20
        plot_h = height - margin_top - 40

        cell_w = plot_w / n_cols
        cell_h = plot_h / n_rows

        max_val = max(1, np.max(cm))

        rects_svg = []
        for i in range(n_rows):
            for j in range(n_cols):
                val = cm[i, j]
                opacity = 0.2 + 0.8 * (val / max_val)
                x = margin_left + j * cell_w
                y = margin_top + i * cell_h

                color = "#3b82f6" if i == j else "#ef4444"
                rects_svg.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-2:.1f}" height="{cell_h-2:.1f}" fill="{color}" opacity="{opacity:.2f}" rx="4"/>'
                )
                rects_svg.append(
                    f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 5:.1f}" fill="#ffffff" font-weight="bold" font-size="14" text-anchor="middle">{val}</text>'
                )

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <text x="20" y="25" fill="#f8fafc" font-size="13" font-weight="bold">Confusion Matrix</text>
        {''.join(rects_svg)}
    </svg>"""
        return svg

    @staticmethod
    def generate_roc_curve_svg(
        fpr: np.ndarray, tpr: np.ndarray, auc_score: float = 0.5, width: int = 500, height: int = 300
    ) -> str:
        """Generate inline SVG string of ROC Curve."""
        if len(fpr) == 0 or len(tpr) == 0:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No ROC Data</text></svg>'

        margin_left = 50
        margin_right = 20
        margin_top = 30
        margin_bottom = 40
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        pts = []
        for f, t in zip(fpr, tpr):
            x = margin_left + float(f) * plot_w
            y = margin_top + (1.0 - float(t)) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")

        polyline_pts = " ".join(pts)

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <text x="20" y="20" fill="#f8fafc" font-size="13" font-weight="bold">ROC Curve (AUC = {auc_score:.3f})</text>
        <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#475569" stroke-width="1"/>
        <!-- Diagonal Baseline -->
        <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{margin_top}" stroke="#64748b" stroke-dasharray="4" stroke-width="1"/>
        <!-- ROC Curve Polyline -->
        <polyline points="{polyline_pts}" fill="none" stroke="#10b981" stroke-width="2.5"/>
    </svg>"""
        return svg

    @staticmethod
    def generate_feature_importance_svg(
        importance_series: pd.Series, top_n: int = 10, width: int = 600, height: int = 300
    ) -> str:
        """Generate inline SVG horizontal bar chart of top N feature importances."""
        if importance_series.empty:
            return f'<svg width="{width}" height="{height}"><text x="20" y="50" fill="#94a3b8">No Importance Data</text></svg>'

        top_series = importance_series.head(top_n)
        max_val = max(1e-6, float(top_series.max()))
        n_items = len(top_series)

        margin_left = 140
        margin_right = 20
        margin_top = 30
        margin_bottom = 20
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        bar_h = max(6, (plot_h / n_items) - 4)

        bars_svg = []
        for i, (name, val) in enumerate(top_series.items()):
            y = margin_top + i * (bar_h + 4)
            w = (val / max_val) * plot_w
            bars_svg.append(f'<text x="{margin_left - 10}" y="{y + bar_h - 2:.1f}" fill="#94a3b8" font-size="11" text-anchor="end">{name[:18]}</text>')
            bars_svg.append(f'<rect x="{margin_left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="#3b82f6" rx="2"/>')
            bars_svg.append(f'<text x="{margin_left + w + 5:.1f}" y="{y + bar_h - 2:.1f}" fill="#f8fafc" font-size="10">{val:.3f}</text>')

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#1e293b; border-radius:8px; font-family:sans-serif;">
        <text x="20" y="20" fill="#f8fafc" font-size="13" font-weight="bold">Top Feature Importances</text>
        {''.join(bars_svg)}
    </svg>"""
        return svg
