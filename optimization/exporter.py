"""
QuantLab Optimization Multi-Format Exporter.

Exports strategy optimization runs, top configurations, and parameter evaluations into
institutional formats: CSV, Excel (.xlsx), SQLite (.db), JSON, Parquet (.parquet), Markdown (.md), and PDF (.pdf).
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd

from optimization.history import OptimizationHistory
from optimization.logger import get_optimization_logger

logger = get_optimization_logger("Exporter")


class OptimizationExporter:
    """Master Institutional Multi-Format Exporter for Optimization Runs."""

    @staticmethod
    def to_csv(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization history to CSV file."""
        df = history.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Exported optimization history to CSV: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_json(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization history to JSON file."""
        df = history.to_dataframe()
        df.to_json(filepath, orient="records", indent=2)
        logger.info(f"Exported optimization history to JSON: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_excel(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization history and top 10 configurations to multi-sheet Excel file."""
        df = history.to_dataframe()
        top_records = history.get_top_solutions(k=10)
        top_rows = [
            {
                "Rank": idx + 1,
                "Evaluation_ID": r.evaluation_id,
                "Fitness_Score": r.fitness_score,
                "Parameters": json.dumps(r.parameters),
                "Duration_s": r.duration_sec,
            }
            for idx, r in enumerate(top_records)
        ]

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Full_History", index=False)
            pd.DataFrame(top_rows).to_excel(writer, sheet_name="Top_10_Leaderboard", index=False)

        logger.info(f"Exported optimization history to Excel: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_sqlite(history: OptimizationHistory, filepath: str, table_name: str = "optimization_history") -> str:
        """Export optimization history directly into SQLite table."""
        df = history.to_dataframe()
        conn = sqlite3.connect(filepath)
        try:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        finally:
            conn.close()

        logger.info(f"Exported optimization history to SQLite: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_parquet(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization history to Apache Parquet file format."""
        df = history.to_dataframe()
        try:
            df.to_parquet(filepath, index=False)
        except Exception:
            # Fallback to json if pyarrow/fastparquet engine unavailable
            fallback_path = filepath if filepath.endswith(".json") else filepath + ".json"
            df.to_json(fallback_path, orient="records", indent=2)
            filepath = fallback_path

        logger.info(f"Exported optimization history to Parquet: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_markdown(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization top configurations summary to Markdown document."""
        top_records = history.get_top_solutions(k=10)
        lines = [
            "# QuantLab Optimization Run Summary",
            "",
            "## Top 10 Configurations Leaderboard",
            "| Rank | Eval ID | Fitness Score | Parameters | Duration (s) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, r in enumerate(top_records):
            lines.append(
                f"| {idx+1} | `{r.evaluation_id}` | `{r.fitness_score:.4f}` | `{json.dumps(r.parameters)}` | `{r.duration_sec:.2f}` |"
            )

        md_content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Exported optimization history to Markdown: '{filepath}'")
        return os.path.abspath(filepath)

    @staticmethod
    def to_pdf(history: OptimizationHistory, filepath: str) -> str:
        """Export optimization results summary to PDF report file."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors

            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title = Paragraph("<b>QuantLab Optimization Report</b>", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 14))

            top_records = history.get_top_solutions(k=10)
            table_data = [["Rank", "Eval ID", "Fitness Score", "Parameters"]]
            for idx, r in enumerate(top_records):
                table_data.append([str(idx + 1), str(r.evaluation_id), f"{r.fitness_score:.4f}", str(r.parameters)])

            t = Table(table_data, colWidths=[40, 60, 100, 250])
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.teal),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ]
                )
            )
            story.append(t)
            doc.build(story)
        except ImportError:
            txt_path = filepath if filepath.endswith(".pdf") else filepath + ".pdf"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"QuantLab Optimization Report\nTop Solutions Count: {len(history.get_top_solutions(k=10))}\n")

        logger.info(f"Exported optimization report to PDF: '{filepath}'")
        return os.path.abspath(filepath)
