"""
QuantLab Portfolio Risk Engine & Stress Testing Framework.

Calculates Value at Risk (Parametric, Historical, Monte Carlo VaR), Conditional VaR (Expected Shortfall),
Tail Risk, Beta against benchmark, Tracking Error, Annualized Volatility, and Stress Test scenario shocks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PortfolioRiskMetrics:
    """Dataclass holding institutional portfolio risk telemetry metrics."""

    var_parametric_95: float = 0.0
    var_parametric_99: float = 0.0
    var_historical_95: float = 0.0
    var_historical_99: float = 0.0
    cvar_expected_shortfall_95: float = 0.0
    cvar_expected_shortfall_99: float = 0.0
    tail_ratio: float = 0.0
    volatility_annualized: float = 0.0
    beta: float = 1.0
    tracking_error: float = 0.0
    stress_test_results: Dict[str, float] = field(default_factory=dict)


class RiskEngine:
    """Institutional Portfolio Risk Engine & Stress Tester."""

    @staticmethod
    def compute_parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Parametric Value at Risk (VaR)."""
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        mu = float(clean.mean())
        std = float(clean.std())
        z = stats.norm.ppf(1.0 - confidence)
        return float(-(mu + z * std))

    @staticmethod
    def compute_historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Historical Value at Risk (VaR)."""
        clean = returns.dropna().values
        if len(clean) == 0:
            return 0.0
        percentile = (1.0 - confidence) * 100.0
        return float(-np.percentile(clean, percentile))

    @staticmethod
    def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Compute Conditional Value at Risk (CVaR / Expected Shortfall)."""
        clean = returns.dropna().values
        if len(clean) == 0:
            return 0.0
        percentile = (1.0 - confidence) * 100.0
        cutoff = np.percentile(clean, percentile)
        tail = clean[clean <= cutoff]
        if len(tail) == 0:
            return float(-cutoff)
        return float(-np.mean(tail))

    @staticmethod
    def compute_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Compute Beta relative to benchmark returns series."""
        df = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(df) < 2:
            return 1.0
        cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1])
        var_bm = float(np.var(df.iloc[:, 1]))
        if var_bm <= 0:
            return 1.0
        return float(cov / var_bm)

    @staticmethod
    def compute_tracking_error(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Compute Tracking Error against benchmark."""
        diff = (returns - benchmark_returns).dropna()
        if len(diff) == 0:
            return 0.0
        return float(diff.std() * np.sqrt(252.0))

    @staticmethod
    def run_stress_test(portfolio_returns: pd.Series) -> Dict[str, float]:
        """Execute stress testing under simulated market shock scenarios.

        Returns:
            Dictionary mapping scenario name to simulated portfolio return impact.
        """
        clean = portfolio_returns.dropna().values
        if len(clean) == 0:
            return {}

        std = float(np.std(clean))
        return {
            "2008_Financial_Crash_Shock": -3.5 * std * np.sqrt(10),
            "2020_COVID_Flash_Crash": -2.8 * std * np.sqrt(5),
            "Interest_Rate_Spike": -2.0 * std * np.sqrt(10),
            "Liquidity_Freeze": -3.0 * std * np.sqrt(7),
        }

    @classmethod
    def analyze_risk(
        cls, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ) -> PortfolioRiskMetrics:
        """Analyze full portfolio risk telemetry.

        Returns:
            PortfolioRiskMetrics instance.
        """
        clean = returns.dropna()
        vol = float(clean.std() * np.sqrt(252.0)) if len(clean) > 0 else 0.0

        v95_p = cls.compute_parametric_var(clean, 0.95)
        v99_p = cls.compute_parametric_var(clean, 0.99)
        v95_h = cls.compute_historical_var(clean, 0.95)
        v99_h = cls.compute_historical_var(clean, 0.99)
        cv95 = cls.compute_cvar(clean, 0.95)
        cv99 = cls.compute_cvar(clean, 0.99)

        beta = cls.compute_beta(clean, benchmark_returns) if benchmark_returns is not None else 1.0
        te = cls.compute_tracking_error(clean, benchmark_returns) if benchmark_returns is not None else 0.0

        # Tail ratio: 95th percentile gain / 5th percentile loss
        if len(clean) > 0:
            p95 = float(np.percentile(clean, 95))
            p5 = float(np.percentile(clean, 5))
            tail_r = abs(p95 / p5) if p5 != 0 else 1.0
        else:
            tail_r = 1.0

        stress = cls.run_stress_test(clean)

        return PortfolioRiskMetrics(
            var_parametric_95=v95_p,
            var_parametric_99=v99_p,
            var_historical_95=v95_h,
            var_historical_99=v99_h,
            cvar_expected_shortfall_95=cv95,
            cvar_expected_shortfall_99=cv99,
            tail_ratio=tail_r,
            volatility_annualized=vol,
            beta=beta,
            tracking_error=te,
            stress_test_results=stress,
        )
