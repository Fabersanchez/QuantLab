"""
QuantLab Master Portfolio Engine.

Centralizes multi-portfolio lifecycle management: creation, editing, duplicating, deleting,
versioning, multi-portfolio benchmarking/comparison, saving, loading, and multi-format exporting.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd

from portfolio.allocation import AllocationEngine
from portfolio.exporter import PortfolioExporter
from portfolio.logger import get_portfolio_logger
from portfolio.optimizer import PortfolioOptimizer
from portfolio.performance import PerformanceAnalyzer
from portfolio.portfolio import Portfolio
from portfolio.rebalancer import PortfolioRebalancer, RebalanceTrigger
from portfolio.reports import PortfolioReportEngine
from portfolio.risk import RiskEngine, PortfolioRiskMetrics
from portfolio.simulator import PortfolioSimulator, PortfolioSimulationResult

logger = get_portfolio_logger("PortfolioEngine")


class PortfolioEngine:
    """Master Institutional Portfolio Engine for QuantLab."""

    def __init__(self) -> None:
        """Initialize PortfolioEngine."""
        self._portfolios: Dict[str, Portfolio] = {}
        self.allocation_engine = AllocationEngine()
        self.optimizer = PortfolioOptimizer()
        self.rebalancer = PortfolioRebalancer()
        self.simulator = PortfolioSimulator()

    def create_portfolio(
        self, name: str, initial_capital: float = 100000.0, description: str = ""
    ) -> Portfolio:
        """Create a new portfolio instance.

        Args:
            name: Portfolio name string.
            initial_capital: Initial capital float.
            description: Optional portfolio description.

        Returns:
            Created Portfolio instance.
        """
        portfolio = Portfolio(name=name, initial_capital=initial_capital, current_cash=initial_capital, description=description)
        self._portfolios[portfolio.portfolio_id] = portfolio
        logger.log_creation(portfolio.portfolio_id, name, initial_capital)
        return portfolio

    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Fetch portfolio by ID."""
        return self._portfolios.get(portfolio_id)

    def list_portfolios(self) -> List[Portfolio]:
        """List all managed portfolio instances."""
        return list(self._portfolios.values())

    def edit_portfolio(
        self, portfolio_id: str, name: Optional[str] = None, initial_capital: Optional[float] = None
    ) -> Optional[Portfolio]:
        """Edit existing portfolio attributes."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return None
        if name:
            portfolio.name = name
        if initial_capital is not None:
            portfolio.initial_capital = initial_capital
        return portfolio

    def duplicate_portfolio(self, portfolio_id: str, new_name: Optional[str] = None) -> Optional[Portfolio]:
        """Duplicate an existing portfolio with new unique ID."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return None
        cloned = portfolio.clone(new_name=new_name)
        self._portfolios[cloned.portfolio_id] = cloned
        return cloned

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """Delete portfolio by ID."""
        if portfolio_id in self._portfolios:
            del self._portfolios[portfolio_id]
            return True
        return False

    def version_portfolio(self, portfolio_id: str, version_type: str = "patch") -> Optional[str]:
        """Increment portfolio semantic version string."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            return None
        return portfolio.increment_version(version_type=version_type)

    def save_portfolio(self, portfolio_id: str, filepath: str) -> str:
        """Save portfolio to JSON or SQLite file."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_id}' not found.")
        if filepath.endswith(".db") or filepath.endswith(".sqlite"):
            portfolio.to_sqlite(filepath)
        else:
            portfolio.to_json(filepath=filepath)
        return filepath

    def load_portfolio(self, filepath: str) -> Portfolio:
        """Load portfolio from JSON file and register in engine."""
        portfolio = Portfolio.from_json(filepath)
        self._portfolios[portfolio.portfolio_id] = portfolio
        return portfolio

    def compare_portfolios(self, portfolio_ids: List[str]) -> pd.DataFrame:
        """Compare multiple portfolios side-by-side.

        Returns:
            DataFrame matrix comparing initial capital, total assets, version, created timestamp.
        """
        records = []
        for pid in portfolio_ids:
            p = self.get_portfolio(pid)
            if p:
                records.append(
                    {
                        "portfolio_id": p.portfolio_id,
                        "name": p.name,
                        "version": p.version,
                        "initial_capital": p.initial_capital,
                        "total_assets": len(p.assets),
                        "created_at": p.created_at,
                    }
                )
        return pd.DataFrame(records)

    def export_portfolio(self, portfolio_id: str, filepath: str, export_format: str = "json") -> str:
        """Export portfolio to specified format (csv, excel, json, sqlite, parquet, markdown, pdf)."""
        portfolio = self.get_portfolio(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_id}' not found.")

        fmt = export_format.lower()
        if fmt == "csv":
            return PortfolioExporter.to_csv(portfolio, filepath)
        elif fmt in ("excel", "xlsx"):
            return PortfolioExporter.to_excel(portfolio, filepath)
        elif fmt in ("sqlite", "db"):
            return PortfolioExporter.to_sqlite(portfolio, filepath)
        elif fmt in ("parquet",):
            return PortfolioExporter.to_parquet(portfolio, filepath)
        elif fmt in ("markdown", "md"):
            return PortfolioExporter.to_markdown(portfolio, filepath)
        elif fmt in ("pdf",):
            return PortfolioExporter.to_pdf(portfolio, filepath)
        else:
            return PortfolioExporter.to_json(portfolio, filepath)
