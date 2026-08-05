"""
QuantLab Master Portfolio Engine & Simulator Package.

Provides institutional portfolio management, 10 capital allocation models, portfolio optimization,
automated rebalancing triggers, exposure analysis, risk engines (VaR/CVaR/Beta), multi-thousand path
simulations, performance telemetry, formal report generation, and multi-format exporters.
"""

from portfolio.allocation import AllocationEngine
from portfolio.allocation_models import (
    BaseAllocationModel,
    BlackLittermanModel,
    CustomAllocationModel,
    EqualRiskContributionModel,
    EqualWeightModel,
    HRPAllocationModel,
    KellyAllocationModel,
    MaximumDiversificationModel,
    MeanVarianceModel,
    MinimumVarianceModel,
    RiskParityModel,
)
from portfolio.asset import Asset, MarketType
from portfolio.constraints import PortfolioConstraints
from portfolio.correlation import CorrelationAnalyzer
from portfolio.covariance import CovarianceAnalyzer
from portfolio.diversification import DiversificationAnalyzer
from portfolio.exporter import PortfolioExporter
from portfolio.exposure import ExposureAnalyzer, PortfolioExposure
from portfolio.logger import PortfolioLogger, get_portfolio_logger
from portfolio.metrics import PortfolioMetrics, PortfolioMetricsResult
from portfolio.optimizer import PortfolioOptimizer
from portfolio.performance import PerformanceAnalyzer
from portfolio.portfolio import Portfolio
from portfolio.portfolio_engine import PortfolioEngine
from portfolio.rebalancer import PortfolioRebalancer, RebalanceTrigger
from portfolio.reports import PortfolioReportEngine
from portfolio.risk import PortfolioRiskMetrics, RiskEngine
from portfolio.simulator import PortfolioSimulationResult, PortfolioSimulator, SimulationConfig

__all__ = [
    "PortfolioEngine",
    "Portfolio",
    "Asset",
    "MarketType",
    "AllocationEngine",
    "BaseAllocationModel",
    "EqualWeightModel",
    "RiskParityModel",
    "MinimumVarianceModel",
    "MaximumDiversificationModel",
    "KellyAllocationModel",
    "HRPAllocationModel",
    "BlackLittermanModel",
    "MeanVarianceModel",
    "EqualRiskContributionModel",
    "CustomAllocationModel",
    "PortfolioOptimizer",
    "PortfolioRebalancer",
    "RebalanceTrigger",
    "ExposureAnalyzer",
    "PortfolioExposure",
    "DiversificationAnalyzer",
    "CorrelationAnalyzer",
    "CovarianceAnalyzer",
    "RiskEngine",
    "PortfolioRiskMetrics",
    "PerformanceAnalyzer",
    "PortfolioConstraints",
    "PortfolioMetrics",
    "PortfolioMetricsResult",
    "PortfolioSimulator",
    "SimulationConfig",
    "PortfolioSimulationResult",
    "PortfolioExporter",
    "PortfolioReportEngine",
    "PortfolioLogger",
    "get_portfolio_logger",
]
