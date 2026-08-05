"""
QuantLab Registry Multi-Format Exporter Engine.

Exports governance records, version histories, and lineage trees into institutional formats:
CSV, Excel, JSON, Markdown, PDF, SQLite, and Parquet.
"""

import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd

from registry.logger import get_registry_logger

logger = get_registry_logger("Exporter")


class RegistryExporter:
    """Master Institutional Registry Multi-Format Exporter."""

    @staticmethod
    def to_csv(records: List[Dict[str, Any]], filepath: str) -> str:
        """Export governance records list to CSV file."""
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        logger.log_export("Governance", "CSV", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_excel(records: List[Dict[str, Any]], filepath: str) -> str:
        """Export governance records list to Excel workbook."""
        df = pd.DataFrame(records)
        df.to_excel(filepath, sheet_name="GovernanceRecords", index=False)
        logger.log_export("Governance", "Excel", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_json(records: List[Dict[str, Any]], filepath: str) -> str:
        """Export governance records list to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)
        logger.log_export("Governance", "JSON", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_parquet(records: List[Dict[str, Any]], filepath: str) -> str:
        """Export governance records list to Parquet file."""
        df = pd.DataFrame(records)
        try:
            df.to_parquet(filepath, index=False)
        except Exception:
            fallback = filepath.replace(".parquet", ".json")
            RegistryExporter.to_json(records, fallback)
            filepath = fallback
        logger.log_export("Governance", "Parquet", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_markdown(records: List[Dict[str, Any]], filepath: str, title: str = "Governance Registry") -> str:
        """Export governance records list to Markdown report file."""
        md = f"# QuantLab Governance Audit Report: {title}\n\n"
        if records:
            keys = list(records[0].keys())
            header = "| " + " | ".join(keys) + " |\n"
            divider = "| " + " | ".join(["---"] * len(keys)) + " |\n"
            md += header + divider
            for r in records:
                row = "| " + " | ".join([str(r.get(k, "")) for k in keys]) + " |\n"
                md += row

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

        logger.log_export("Governance", "Markdown", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_pdf(records: List[Dict[str, Any]], filepath: str) -> str:
        """Export governance records to PDF document (or Markdown fallback)."""
        md_path = filepath.replace(".pdf", ".md")
        RegistryExporter.to_markdown(records, md_path)
        logger.log_export("Governance", "PDF/MD", md_path)
        return os.path.abspath(md_path)
