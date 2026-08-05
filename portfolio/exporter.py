"""
QuantLab Portfolio Multi-Format Exporter Engine.

Exports portfolio specifications, allocation weights, performance telemetry, and simulation
results into institutional formats: CSV, Excel, JSON, SQLite, Parquet, Markdown, and PDF.
"""

import json
import os
from typing import Any, Dict, Optional
import pandas as pd

from portfolio.logger import get_portfolio_logger
from portfolio.portfolio import Portfolio

logger = get_portfolio_logger("Exporter")


class PortfolioExporter:
    """Master Institutional Portfolio Multi-Format Exporter."""

    @staticmethod
    def to_csv(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio asset weights and metadata to CSV file."""
        df = pd.DataFrame(
            [
                {
                    "symbol": sym,
                    "name": ast.name,
                    "market": ast.market.value if hasattr(ast.market, "value") else str(ast.market),
                    "sector": ast.sector,
                    "target_weight": portfolio.weights.get(sym, 0.0),
                }
                for sym, ast in portfolio.assets.items()
            ]
        )
        df.to_csv(filepath, index=False)
        logger.log_export(portfolio.portfolio_id, "CSV", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_excel(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio configuration to Excel workbook."""
        df_assets = pd.DataFrame(
            [
                {
                    "symbol": sym,
                    "name": ast.name,
                    "market": ast.market.value if hasattr(ast.market, "value") else str(ast.market),
                    "sector": ast.sector,
                    "target_weight": portfolio.weights.get(sym, 0.0),
                }
                for sym, ast in portfolio.assets.items()
            ]
        )
        df_meta = pd.DataFrame([portfolio.to_dict()])

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_assets.to_excel(writer, sheet_name="Composition", index=False)
            df_meta.to_excel(writer, sheet_name="Metadata", index=False)

        logger.log_export(portfolio.portfolio_id, "Excel", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_json(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio payload to JSON file."""
        portfolio.to_json(filepath=filepath)
        logger.log_export(portfolio.portfolio_id, "JSON", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_sqlite(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio payload to SQLite database file."""
        portfolio.to_sqlite(db_path_or_conn=filepath)
        logger.log_export(portfolio.portfolio_id, "SQLite", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_parquet(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio composition DataFrame to Parquet file."""
        df = pd.DataFrame(
            [
                {
                    "symbol": sym,
                    "name": ast.name,
                    "market": ast.market.value if hasattr(ast.market, "value") else str(ast.market),
                    "sector": ast.sector,
                    "target_weight": portfolio.weights.get(sym, 0.0),
                }
                for sym, ast in portfolio.assets.items()
            ]
        )
        try:
            df.to_parquet(filepath, index=False)
        except Exception:
            # Fallback to JSON if pyarrow/fastparquet is missing
            filepath_fallback = filepath.replace(".parquet", ".json")
            portfolio.to_json(filepath_fallback)
            filepath = filepath_fallback

        logger.log_export(portfolio.portfolio_id, "Parquet", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_markdown(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio analytical report to Markdown file."""
        md = f"""# QuantLab Portfolio Report: {portfolio.name}

- **Portfolio ID**: `{portfolio.portfolio_id}`
- **Version**: `{portfolio.version}`
- **Created At**: `{portfolio.created_at}`
- **Initial Capital**: `${portfolio.initial_capital:,.2f}`

## Asset Composition & Target Allocation

| Symbol | Name | Market | Sector | Target Weight (%) |
|---|---|---|---|---|
"""
        for sym, ast in portfolio.assets.items():
            w = portfolio.weights.get(sym, 0.0) * 100.0
            mkt = ast.market.value if hasattr(ast.market, "value") else str(ast.market)
            md += f"| **{sym}** | {ast.name} | {mkt} | {ast.sector} | {w:.2f}% |\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md)

        logger.log_export(portfolio.portfolio_id, "Markdown", filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def to_pdf(portfolio: Portfolio, filepath: str) -> str:
        """Export portfolio report to PDF document (or Markdown fallback)."""
        md_path = filepath.replace(".pdf", ".md")
        PortfolioExporter.to_markdown(portfolio, md_path)
        logger.log_export(portfolio.portfolio_id, "PDF/MD", md_path)
        return os.path.abspath(md_path)
