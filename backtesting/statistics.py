"""
QuantLab Statistical and Distributional Analysis.

Provides deep statistical inspection of backtest outputs including return distribution skewness,
kurtosis, monthly seasonality return tables, holding duration metrics, long vs short performance breakdown,
and Value-at-Risk (VaR / CVaR) calculations.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BacktestStatistics:
    """Institutional Backtest Statistical Analyzer."""

    @staticmethod
    def calculate_trade_distribution(trade_df: pd.DataFrame) -> Dict[str, float]:
        """Compute statistical moments of trade net PnL distribution.

        Args:
            trade_df: DataFrame containing 'net_pnl'.

        Returns:
            Dict containing mean, std, median, skewness, kurtosis, and variance.
        """
        if trade_df.empty or "net_pnl" not in trade_df.columns or len(trade_df) < 2:
            return {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "variance": 0.0,
            }

        pnl = trade_df["net_pnl"]
        return {
            "mean": float(pnl.mean()),
            "std": float(pnl.std()),
            "median": float(pnl.median()),
            "skewness": float(pnl.skew()),
            "kurtosis": float(pnl.kurtosis()),
            "variance": float(pnl.var()),
        }

    @staticmethod
    def generate_seasonality_matrix(equity_df: pd.DataFrame) -> pd.DataFrame:
        """Generate Monthly vs Yearly returns table (seasonality matrix).

        Args:
            equity_df: DataFrame indexed by timestamp or containing 'timestamp' column, with 'equity'.

        Returns:
            pd.DataFrame containing years as rows, months (Jan..Dec) as columns, and total Year return.
        """
        if equity_df.empty or "equity" not in equity_df.columns:
            return pd.DataFrame()

        df = equity_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                return pd.DataFrame()

        # Resample monthly return
        monthly_eq = df["equity"].resample("ME").last()
        monthly_ret = monthly_eq.pct_change()

        # Handle first month return
        if not monthly_eq.empty:
            first_val = monthly_eq.iloc[0]
            start_val = df["equity"].iloc[0]
            if start_val > 0:
                monthly_ret.iloc[0] = (first_val - start_val) / start_val

        records: Dict[int, Dict[str, float]] = {}
        for dt, val in monthly_ret.items():
            year = dt.year
            month_name = dt.strftime("%b")
            if year not in records:
                records[year] = {m: 0.0 for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
            records[year][month_name] = float(val * 100.0) if not np.isnan(val) else 0.0

        res_df = pd.DataFrame.from_dict(records, orient="index")

        # Add total yearly return column
        yearly_eq = df["equity"].resample("YE").last()
        yearly_ret = yearly_eq.pct_change()
        if not yearly_eq.empty:
            yearly_ret.iloc[0] = (yearly_eq.iloc[0] - df["equity"].iloc[0]) / df["equity"].iloc[0]

        year_totals = {}
        for dt, val in yearly_ret.items():
            year_totals[dt.year] = float(val * 100.0) if not np.isnan(val) else 0.0

        res_df["Year_Total"] = res_df.index.map(year_totals).fillna(0.0)
        return res_df

    @staticmethod
    def calculate_holding_duration_stats(trade_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute trade holding duration statistics for winning vs losing trades."""
        if trade_df.empty or "holding_bars" not in trade_df.columns:
            return {
                "avg_holding_bars": 0.0,
                "avg_winning_bars": 0.0,
                "avg_losing_bars": 0.0,
                "max_holding_bars": 0,
                "min_holding_bars": 0,
            }

        wins = trade_df[trade_df["net_pnl"] > 0]
        losses = trade_df[trade_df["net_pnl"] <= 0]

        return {
            "avg_holding_bars": float(trade_df["holding_bars"].mean()),
            "avg_winning_bars": float(wins["holding_bars"].mean()) if not wins.empty else 0.0,
            "avg_losing_bars": float(losses["holding_bars"].mean()) if not losses.empty else 0.0,
            "max_holding_bars": int(trade_df["holding_bars"].max()),
            "min_holding_bars": int(trade_df["holding_bars"].min()),
        }

    @staticmethod
    def calculate_long_vs_short_breakdown(trade_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute performance breakdown separated by LONG vs SHORT trades."""
        if trade_df.empty or "side" not in trade_df.columns:
            return {}

        longs = trade_df[trade_df["side"].str.upper() == "LONG"]
        shorts = trade_df[trade_df["side"].str.upper() == "SHORT"]

        def _side_stats(df_side: pd.DataFrame) -> Dict[str, float]:
            if df_side.empty:
                return {"count": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
            count = len(df_side)
            net_pnl = float(df_side["net_pnl"].sum())
            wins = df_side[df_side["net_pnl"] > 0]
            win_rate = (len(wins) / count) * 100.0
            gross_p = wins["net_pnl"].sum() if not wins.empty else 0.0
            gross_l = abs(df_side[df_side["net_pnl"] <= 0]["net_pnl"].sum())
            pf = (gross_p / gross_l) if gross_l > 0 else (999.0 if gross_p > 0 else 0.0)
            return {"count": count, "net_pnl": net_pnl, "win_rate": win_rate, "profit_factor": pf}

        return {
            "long": _side_stats(longs),
            "short": _side_stats(shorts),
        }

    @staticmethod
    def calculate_value_at_risk(
        returns: pd.Series, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Compute Value at Risk (VaR 95%, 99%) and Conditional VaR / Expected Shortfall.

        Args:
            returns: Equity percentage returns series.
            confidence_level: VaR confidence decimal (e.g., 0.95 for 95%).

        Returns:
            Dict containing historical_var, parametric_var, and cvar_expected_shortfall.
        """
        if returns.empty or len(returns) < 5:
            return {
                "historical_var_pct": 0.0,
                "parametric_var_pct": 0.0,
                "cvar_expected_shortfall_pct": 0.0,
            }

        alpha = 1.0 - confidence_level

        # Historical VaR
        hist_var = float(np.percentile(returns, alpha * 100.0))

        # Parametric Gaussian VaR
        mean = float(returns.mean())
        std = float(returns.std())
        from scipy.stats import norm
        z_score = norm.ppf(alpha)
        parametric_var = mean + (z_score * std)

        # Conditional VaR (Expected Shortfall)
        tail_returns = returns[returns <= hist_var]
        cvar = float(tail_returns.mean()) if not tail_returns.empty else hist_var

        return {
            "historical_var_pct": abs(hist_var) * 100.0,
            "parametric_var_pct": abs(parametric_var) * 100.0,
            "cvar_expected_shortfall_pct": abs(cvar) * 100.0,
        }
