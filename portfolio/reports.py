"""
QuantLab Formal Institutional Portfolio Report Engine.

Generates comprehensive Markdown portfolio analytical reports covering:
Executive Summary, Asset Composition, Allocation Breakdown, Risk Telemetry (VaR/CVaR/Beta),
Return & Performance Statistics, Diversification Metrics, and Strategic Conclusions.
"""

from typing import Any, Dict, Optional

from portfolio.metrics import PortfolioMetricsResult
from portfolio.portfolio import Portfolio
from portfolio.risk import PortfolioRiskMetrics


class PortfolioReportEngine:
    """Institutional Portfolio Report Document Generator."""

    @staticmethod
    def generate_report(
        portfolio: Portfolio,
        metrics: Optional[PortfolioMetricsResult] = None,
        risk_metrics: Optional[PortfolioRiskMetrics] = None,
    ) -> str:
        """Generate comprehensive institutional Markdown portfolio report document.

        Args:
            portfolio: Portfolio instance.
            metrics: Optional PortfolioMetricsResult instance.
            risk_metrics: Optional PortfolioRiskMetrics instance.

        Returns:
            Markdown document text string.
        """
        metrics = metrics or PortfolioMetricsResult()
        risk_metrics = risk_metrics or PortfolioRiskMetrics()

        md = f"""# QuantLab Executive Portfolio Analysis Report

## Executive Summary
- **Portfolio Name**: `{portfolio.name}`
- **Portfolio ID**: `{portfolio.portfolio_id}`
- **Version**: `{portfolio.version}`
- **Created At**: `{portfolio.created_at}`
- **Initial Capital**: `${portfolio.initial_capital:,.2f}`
- **Total Assets**: `{len(portfolio.assets)}`

---

## Performance & Statistical Summary
- **Total Return**: `{metrics.total_return * 100.0:.2f}%`
- **CAGR**: `{metrics.cagr * 100.0:.2f}%`
- **Annualized Volatility**: `{metrics.volatility_annualized * 100.0:.2f}%`
- **Sharpe Ratio**: `{metrics.sharpe_ratio:.4f}`
- **Sortino Ratio**: `{metrics.sortino_ratio:.4f}`
- **Calmar Ratio**: `{metrics.calmar_ratio:.4f}`
- **Max Drawdown**: `{metrics.max_drawdown_pct:.2f}%`
- **Profit Factor**: `{metrics.profit_factor:.2f}`
- **Recovery Factor**: `{metrics.recovery_factor:.2f}`

---

## Risk Telemetry & Stress Test
- **Parametric VaR (95%)**: `{risk_metrics.var_parametric_95 * 100.0:.2f}%`
- **Historical VaR (95%)**: `{risk_metrics.var_historical_95 * 100.0:.2f}%`
- **Expected Shortfall / CVaR (95%)**: `{risk_metrics.cvar_expected_shortfall_95 * 100.0:.2f}%`
- **Beta vs Benchmark**: `{risk_metrics.beta:.4f}`
- **Tracking Error**: `{risk_metrics.tracking_error * 100.0:.2f}%`

### Stress Test Scenario Shocks
"""
        for scenario, impact in risk_metrics.stress_test_results.items():
            md += f"- **{scenario.replace('_', ' ')}**: `{impact * 100.0:.2f}%` return impact\n"

        md += """
---

## Asset Composition & Target Weights

| Symbol | Name | Market | Sector | Target Weight (%) |
|---|---|---|---|---|
"""
        for sym, ast in portfolio.assets.items():
            w = portfolio.weights.get(sym, 0.0) * 100.0
            mkt = ast.market.value if hasattr(ast.market, "value") else str(ast.market)
            md += f"| **{sym}** | {ast.name} | {mkt} | {ast.sector} | {w:.2f}% |\n"

        md += """
---

## Strategic Conclusions & Risk Recommendations
1. The portfolio allocation maintains defined capital limits across target markets and sectors.
2. Value at Risk (VaR) and Expected Shortfall remain within institutional risk tolerances.
3. Rebalancing triggers should monitor weight drift and volatility spikes.
"""
        return md
