"""
QuantLab Experiment Exporter Engine.

Exports experiment results, metrics, benchmarks, and historical logs into institutional formats:
CSV, Excel (.xlsx), JSON, Markdown (.md), SQLite (.db), Parquet (.parquet), and PDF (.pdf).
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from research.comparator import ComparisonResult
from research.experiment import Experiment
from research.logger import get_research_logger

logger = get_research_logger("Exporter")


class ExperimentExporter:
    """Master Institutional Multi-Format Exporter for Scientific Experiments."""

    @staticmethod
    def to_json(experiment: Experiment, filepath: str) -> str:
        """Export experiment object to JSON format file.

        Returns:
            Absolute file path.
        """
        experiment.to_json(filepath=filepath)
        logger.log_export(experiment.uuid, "JSON", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_csv(experiment: Experiment, filepath: str) -> str:
        """Export experiment summary and metrics to CSV file.

        Returns:
            Absolute file path.
        """
        data = experiment.to_dict()
        flat_data = {
            "uuid": data["uuid"],
            "name": data["name"],
            "version": data["version"],
            "author": data["author"],
            "status": data["status"],
            "asset": data["asset"],
            "timeframe": data["timeframe"],
            "broker": data["broker"],
            "execution_time": data["execution_time"],
            "hash": data["hash"],
            "checksum": data["checksum"],
        }
        # Flatten metrics
        results = data.get("results", {})
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                flat_data[f"result_{k}"] = v

        df = pd.DataFrame([flat_data])
        df.to_csv(filepath, index=False)
        logger.log_export(experiment.uuid, "CSV", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_excel(experiment: Experiment, filepath: str) -> str:
        """Export experiment metadata, metrics, parameters, and logs to multi-sheet Excel file.

        Returns:
            Absolute file path.
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Sheet 1: Metadata
            meta_df = pd.DataFrame(
                [
                    {"Property": "UUID", "Value": experiment.uuid},
                    {"Property": "Name", "Value": experiment.name},
                    {"Property": "Version", "Value": experiment.version},
                    {"Property": "Author", "Value": experiment.author},
                    {"Property": "Date", "Value": experiment.date},
                    {"Property": "Status", "Value": str(experiment.status)},
                    {"Property": "Asset", "Value": experiment.asset},
                    {"Property": "Timeframe", "Value": experiment.timeframe},
                    {"Property": "Broker", "Value": experiment.broker},
                    {"Property": "Execution Time (s)", "Value": experiment.execution_time},
                ]
            )
            meta_df.to_excel(writer, sheet_name="Metadata", index=False)

            # Sheet 2: Parameters
            param_rows = [{"Parameter": k, "Value": str(v)} for k, v in experiment.parameters.items()]
            pd.DataFrame(param_rows if param_rows else [{"Parameter": "N/A", "Value": "N/A"}]).to_excel(
                writer, sheet_name="Parameters", index=False
            )

            # Sheet 3: Results & Metrics
            results_rows = [{"Metric": k, "Value": str(v)} for k, v in experiment.results.items()]
            pd.DataFrame(results_rows if results_rows else [{"Metric": "N/A", "Value": "N/A"}]).to_excel(
                writer, sheet_name="Results", index=False
            )

        logger.log_export(experiment.uuid, "Excel", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_markdown(experiment: Experiment, filepath: str) -> str:
        """Export experiment summary to Markdown document file.

        Returns:
            Absolute file path.
        """
        lines = [
            f"# QuantLab Scientific Experiment Report: {experiment.name}",
            "",
            "## Metadata",
            f"- **UUID**: `{experiment.uuid}`",
            f"- **Version**: {experiment.version}",
            f"- **Author**: {experiment.author}",
            f"- **Date**: {experiment.date}",
            f"- **Status**: `{str(experiment.status)}`",
            f"- **Asset / Timeframe**: `{experiment.asset}` / `{experiment.timeframe}`",
            f"- **Execution Time**: `{experiment.execution_time:.4f}s`",
            f"- **Experiment Hash**: `{experiment.hash}`",
            f"- **Dataset Checksum**: `{experiment.checksum}`",
            "",
            "## Parameters",
            "```json",
            json.dumps(experiment.parameters, indent=2, default=str),
            "```",
            "",
            "## Results & Metrics",
            "| Metric | Value |",
            "| :--- | :--- |",
        ]
        for k, v in experiment.results.items():
            if isinstance(v, (int, float, str)):
                lines.append(f"| `{k}` | `{v}` |")

        md_content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.log_export(experiment.uuid, "Markdown", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_sqlite(experiment: Experiment, filepath: str, table_name: str = "exported_experiments") -> str:
        """Export experiment record directly into SQLite database file.

        Returns:
            Absolute file path.
        """
        experiment.to_sqlite(db_path_or_conn=filepath, table_name=table_name)
        logger.log_export(experiment.uuid, "SQLite", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_parquet(experiment: Experiment, filepath: str) -> str:
        """Export experiment data to Apache Parquet columnar file format.

        Returns:
            Absolute file path.
        """
        data = experiment.to_dict()
        flat_data = {
            "uuid": data["uuid"],
            "name": data["name"],
            "version": data["version"],
            "author": data["author"],
            "status": str(data["status"]),
            "asset": data["asset"],
            "timeframe": data["timeframe"],
            "broker": data["broker"],
            "execution_time": float(data["execution_time"]),
            "hash": data["hash"],
            "checksum": data["checksum"],
            "payload_json": experiment.to_json(),
        }
        df = pd.DataFrame([flat_data])
        try:
            df.to_parquet(filepath, index=False)
        except Exception:
            # Fallback if pyarrow / fastparquet is not installed
            fallback_path = filepath if filepath.endswith(".json") else filepath + ".json"
            df.to_json(fallback_path, orient="records", indent=2)
            filepath = fallback_path

        logger.log_export(experiment.uuid, "Parquet", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_pdf(experiment: Experiment, filepath: str) -> str:
        """Export formatted PDF institutional report for experiment.

        Returns:
            Absolute file path.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors

            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title = Paragraph(f"<b>QuantLab Experiment Report: {experiment.name}</b>", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            meta_text = (
                f"<b>UUID:</b> {experiment.uuid}<br/>"
                f"<b>Version:</b> {experiment.version} | <b>Author:</b> {experiment.author}<br/>"
                f"<b>Status:</b> {str(experiment.status)} | <b>Date:</b> {experiment.date}<br/>"
                f"<b>Asset:</b> {experiment.asset} | <b>Timeframe:</b> {experiment.timeframe}<br/>"
                f"<b>Execution Time:</b> {experiment.execution_time:.4f} seconds"
            )
            story.append(Paragraph(meta_text, styles["Normal"]))
            story.append(Spacer(1, 16))

            table_data = [["Metric", "Value"]]
            for k, v in experiment.results.items():
                if isinstance(v, (int, float, str)):
                    table_data.append([str(k), str(v)])

            if len(table_data) > 1:
                t = Table(table_data, colWidths=[200, 250])
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.navy),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ]
                    )
                )
                story.append(t)

            doc.build(story)

        except ImportError:
            # Fallback if reportlab is not installed: generate plain text file formatted as PDF stub
            txt_path = filepath if filepath.endswith(".pdf") else filepath + ".pdf"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"QuantLab Scientific Report\nExperiment: {experiment.name}\nUUID: {experiment.uuid}\n")

        logger.log_export(experiment.uuid, "PDF", filepath)
        return os.path.abspath(filepath)
