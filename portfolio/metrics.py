"""
QuantLab Comprehensive Portfolio Metrics Engine.

Calculates 20+ institutional metrics:
Total Return, CAGR, Monthly/Daily Return, Sharpe, Sortino, Calmar, Treynor Ratio, Information Ratio,
Alpha, Beta, Omega Ratio, MAR Ratio, Profit Factor, Recovery Factor, Ulcer Index, Skewness, Kurtosis, Tail Ratio.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PortfolioMetricsResult:
    """Dataclass holding complete portfolio performance & statistical metrics payload."""

    total_return: float = 0.0
    cagr: float = 0.0
    daily_return_mean: float = 0.0
    volatility_annualized: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    alpha: float = 0.0
    beta: float = 1.0
    omega_ratio: float = 0.0
    mar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0


class PortfolioMetrics:
    """Institutional Portfolio Metrics Engine."""

    @classmethod
    def calculate(
        cls,
        equity_series: pd.Series,
        benchmark_series: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> PortfolioMetricsResult:
        """Calculate comprehensive portfolio performance & risk metrics payload.

        Args:
            equity_series: Series of equity values.
            benchmark_series: Optional benchmark equity series.
            risk_free_rate: Risk-free rate float.

        Returns:
            PortfolioMetricsResult instance.
        """
        if len(equity_series) < 2:
            return PortfolioMetricsResult()

        returns = equity_series.pct_change().dropna()
        start_eq = float(equity_series.iloc[0])
        end_eq = float(equity_series.iloc[-1])
        tot_ret = (end_eq - start_eq) / start_eq if start_eq > 0 else 0.0

        n_bars = len(equity_series)
        years = n_bars / 252.0
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0 if (years > 0 and start_eq > 0 and end_eq > 0) else 0.0

        daily_mean = float(returns.mean())
        vol = float(returns.std() * np.sqrt(252.0))

        # Sharpe ratio
        sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = float(downside.std() * np.sqrt(252.0)) if len(downside) > 0 else 1e-4
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0.0

        # Max Drawdown
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        max_dd = float(abs(dd.min()))

        # Calmar & MAR ratios
        calmar = cagr / max_dd if max_dd > 0 else 0.0
        mar = calmar

        # Profit Factor & Recovery Factor
        pos_ret = returns[returns > 0].sum()
        neg_ret = abs(returns[returns < 0].sum())
        pf = float(pos_ret / neg_ret) if neg_ret > 0 else (10.0 if pos_ret > 0 else 0.0)
        rec_factor = float(tot_ret / max_dd) if max_dd > 0 else 0.0

        # Ulcer Index
        squared_dd = (dd * 100.0) ** 2
        ulcer = float(np.sqrt(squared_dd.mean())) if len(squared_dd) > 0 else 0.0

        # Skewness & Kurtosis
        skew = float(stats.skew(returns.values)) if len(returns) > 2 else 0.0
        kurt = float(stats.kurtosis(returns.values)) if len(returns) > 3 else 0.0

        # Tail Ratio
        if len(returns) > 0:
            p95 = float(np.percentile(returns, 95))
            p5 = float(np.percentile(returns, 5))
            tail_r = abs(p95 / p5) if p5 != 0 else 1.0
        else:
            tail_r = 1.0

        # Omega ratio
        threshold = risk_free_rate / 252.0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        sum_gains = float(gains.sum())
        sum_losses = float(losses.sum())
        omega = sum_gains / sum_losses if sum_losses > 0 else 1.0

        # Benchmark metrics (Alpha, Beta, Treynor, Information Ratio)
        alpha, beta, treynor, info_ratio = 0.0, 1.0, 0.0, 0.0
        if benchmark_series is not None and len(benchmark_series) >= len(equity_series):
            bm_returns = benchmark_series.pct_change().dropna()
            df = pd.concat([returns, bm_returns], axis=1).dropna()
            if len(df) > 1:
                cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1])
                var_bm = float(np.var(df.iloc[:, 1]))
                beta = cov / var_bm if var_bm > 0 else 1.0

                bm_cagr = (float(benchmark_series.iloc[-1]) / float(benchmark_series.iloc[0])) ** (1.0 / years) - 1.0 if years > 0 else 0.0
                alpha = cagr - (risk_free_rate + beta * (bm_cagr - risk_free_rate))
                treynor = (cagr - risk_free_rate) / beta if beta != 0 else 0.0

                diff = returns - bm_returns
                te = float(diff.std() * np.sqrt(252.0))
                info_ratio = (cagr - bm_cagr) / te if te > 0 else 0.0

        return PortfolioMetricsResult(
            total_return=tot_ret,
            cagr=cagr,
            daily_return_mean=daily_mean,
            volatility_annualized=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            treynor_ratio=treynor,
            information_ratio=info_ratio,
            alpha=alpha,
            beta=beta,
            omega_ratio=omega,
            mar_ratio=mar,
            max_drawdown_pct=max_dd * 100.0,
            profit_factor=pf,
            recovery_factor=rec_factor,
            ulcer_index=ulcer,
            skewness=skew,
            kurtosis=kurt,
            tail_ratio=tail_r,
        )
