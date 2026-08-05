"""
QuantLab Portfolio Performance Analyzer.

Calculates cumulative returns, CAGR, annualized volatility, monthly returns matrices,
equity curve series, underwater drawdown series, and benchmark relative metrics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """Institutional Portfolio Performance Analyzer."""

    @staticmethod
    def compute_cumulative_return(equity_series: pd.Series) -> float:
        """Compute cumulative portfolio return fraction."""
        if len(equity_series) < 2:
            return 0.0
        start = float(equity_series.iloc[0])
        end = float(equity_series.iloc[-1])
        if start <= 0:
            return 0.0
        return (end - start) / start

    @staticmethod
    def compute_cagr(equity_series: pd.Series, periods_per_year: float = 252.0) -> float:
        """Compute Compound Annual Growth Rate (CAGR)."""
        if len(equity_series) < 2:
            return 0.0
        start = float(equity_series.iloc[0])
        end = float(equity_series.iloc[-1])
        if start <= 0 or end <= 0:
            return 0.0
        years = len(equity_series) / periods_per_year
        if years <= 0:
            return 0.0
        return float((end / start) ** (1.0 / years) - 1.0)

    @staticmethod
    def compute_monthly_returns(returns_series: pd.Series) -> pd.DataFrame:
        """Convert daily returns series into Year x Month matrix DataFrame.

        Returns:
            DataFrame indexed by Year with Month columns (Jan..Dec).
        """
        if returns_series.empty or not isinstance(returns_series.index, pd.DatetimeIndex):
            return pd.DataFrame()

        df = returns_series.to_frame(name="return")
        df["year"] = df.index.year
        df["month"] = df.index.month

        pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc=lambda x: (1.0 + x).prod() - 1.0)
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]
        return pivot * 100.0
