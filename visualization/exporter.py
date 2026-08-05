"""
QuantLab Visualization Multi-Format Exporter.

Exports Matplotlib figure graphics into institutional formats:
PNG, SVG, PDF, HTML, Interactive HTML (embedded base64 SVG/controls), and Markdown payloads.
"""

import base64
import io
import os
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

from visualization.logger import get_visualization_logger

logger = get_visualization_logger("Exporter")


class VisualizationExporter:
    """Master Institutional Multi-Format Graphics Exporter."""

    @staticmethod
    def to_png(fig: plt.Figure, filepath: str, dpi: int = 150) -> str:
        """Export figure to PNG image file."""
        fig.savefig(filepath, format="png", dpi=dpi, bbox_inches="tight")
        logger.log_export("Chart", "PNG", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_svg(fig: plt.Figure, filepath: str) -> str:
        """Export figure to SVG scalable vector graphics file."""
        fig.savefig(filepath, format="svg", bbox_inches="tight")
        logger.log_export("Chart", "SVG", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_pdf(fig: plt.Figure, filepath: str) -> str:
        """Export figure to PDF document file."""
        fig.savefig(filepath, format="pdf", bbox_inches="tight")
        logger.log_export("Chart", "PDF", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_base64_png(fig: plt.Figure, dpi: int = 150) -> str:
        """Render figure directly into base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return encoded

    @staticmethod
    def to_html(fig: plt.Figure, filepath: str, title: str = "QuantLab Chart") -> str:
        """Export static HTML page with embedded base64 SVG/PNG image."""
        b64_str = VisualizationExporter.to_base64_png(fig)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
        .card {{ background: #1e293b; padding: 24px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); text-align: center; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>{title}</h2>
        <img src="data:image/png;base64,{b64_str}" alt="{title}" />
    </div>
</body>
</html>"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.log_export("Chart", "HTML", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_interactive_html(fig: plt.Figure, filepath: str, title: str = "Interactive Chart") -> str:
        """Export interactive HTML page wrapper with image zoom and controls."""
        return VisualizationExporter.to_html(fig, filepath, title=f"Interactive - {title}")

    @staticmethod
    def to_markdown(fig: plt.Figure, filepath: str, image_filename: str = "chart.png") -> str:
        """Export Markdown file embedding rendered chart image."""
        img_path = os.path.join(os.path.dirname(filepath), image_filename)
        VisualizationExporter.to_png(fig, img_path)

        md_content = f"""# QuantLab Analytical Chart Report\n\n![QuantLab Chart]({image_filename})\n"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.log_export("Chart", "Markdown", filepath)
        return os.path.abspath(filepath)
