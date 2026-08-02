"""
QuantLab Institutional Performance Metrics.

Calculates comprehensive quantitative performance indicators: Profit Factor, Win Rate,
Sharpe Ratio, Sortino Ratio, Calmar Ratio, Ulcer Index, Expectancy, Recovery Factor,
Average Trade, Max Consecutive Wins & Losses, CAGR, Payoff Ratio, and Drawdowns.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Quantitative Backtest Performance Metrics Calculator."""

    @staticmethod
    def calculate_all(
        equity_df: pd.DataFrame,
        trade_df: pd.DataFrame,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """Calculate complete suite of performance metrics.

        Args:
            equity_df: DataFrame containing 'equity', 'drawdown_pct', 'drawdown_amount'.
            trade_df: DataFrame containing trade logs ('net_pnl', 'pnl_pct', 'is_win').
            initial_capital: Account starting capital.
            risk_free_rate: Annualized risk-free rate decimal (e.g., 0.02 for 2%).
            periods_per_year: Trading periods per year (252 daily, 8760 hourly).

        Returns:
            Dictionary containing all calculated metrics.
        """
        # Equity metrics
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty and "equity" in equity_df.columns else initial_capital
        net_profit = final_equity - initial_capital
        total_return_pct = (net_profit / initial_capital) * 100.0 if initial_capital > 0 else 0.0

        # Drawdown metrics
        max_dd_amount = equity_df["drawdown_amount"].max() if not equity_df.empty and "drawdown_amount" in equity_df.columns else 0.0
        max_dd_pct = equity_df["drawdown_pct"].max() if not equity_df.empty and "drawdown_pct" in equity_df.columns else 0.0

        # Returns series
        if not equity_df.empty and "equity" in equity_df.columns:
            returns = equity_df["equity"].pct_change().dropna()
        else:
            returns = pd.Series(dtype=float)

        # CAGR calculation
        cagr = PerformanceMetrics.calculate_cagr(equity_df, initial_capital, periods_per_year)

        # Ratios
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        calmar = PerformanceMetrics.calculate_calmar_ratio(cagr, max_dd_pct)
        ulcer_index = PerformanceMetrics.calculate_ulcer_index(equity_df)

        # Trade-based statistics
        total_trades = len(trade_df)
        if total_trades > 0 and "net_pnl" in trade_df.columns:
            wins = trade_df[trade_df["net_pnl"] > 0]
            losses = trade_df[trade_df["net_pnl"] <= 0]
            winning_trades = len(wins)
            losing_trades = len(losses)
            win_rate = (winning_trades / total_trades) * 100.0

            gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
            gross_loss = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

            avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
            avg_loss = abs(losses["net_pnl"].mean()) if not losses.empty else 0.0
            payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)

            avg_trade = trade_df["net_pnl"].mean()
            avg_trade_pct = trade_df["pnl_pct"].mean() if "pnl_pct" in trade_df.columns else 0.0

            expectancy = (win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss)
            recovery_factor = (net_profit / max_dd_amount) if max_dd_amount > 0 else (999.0 if net_profit > 0 else 0.0)

            max_consec_wins, max_consec_losses = PerformanceMetrics.calculate_consecutive_streaks(trade_df)
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            gross_profit = 0.0
            gross_loss = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            payoff_ratio = 0.0
            avg_trade = 0.0
            avg_trade_pct = 0.0
            expectancy = 0.0
            recovery_factor = 0.0
            max_consec_wins = 0
            max_consec_losses = 0

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "net_profit": net_profit,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "max_drawdown_amount": max_dd_amount,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "ulcer_index": ulcer_index,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "recovery_factor": recovery_factor,
            "average_trade": avg_trade,
            "average_trade_pct": avg_trade_pct,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "payoff_ratio": payoff_ratio,
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
        }

    @staticmethod
    def calculate_cagr(
        equity_df: pd.DataFrame, initial_capital: float, periods_per_year: int = 252
    ) -> float:
        """Calculate Compound Annual Growth Rate (CAGR) decimal."""
        if equity_df.empty or len(equity_df) < 2 or initial_capital <= 0:
            return 0.0

        final_eq = float(equity_df["equity"].iloc[-1])
        if final_eq <= 0:
            return -1.0

        total_bars = len(equity_df)
        years = total_bars / periods_per_year
        if years <= 0:
            return 0.0

        return (final_eq / initial_capital) ** (1.0 / years) - 1.0

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sharpe Ratio."""
        if returns.empty or len(returns) < 2:
            return 0.0

        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        std = excess_returns.std()
        if std == 0 or np.isnan(std):
            return 0.0

        mean = excess_returns.mean()
        return (mean / std) * math.sqrt(periods_per_year)

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """Calculate annualized Sortino Ratio (downside risk metric)."""
        if returns.empty or len(returns) < 2:
            return 0.0

        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        downside_returns = excess_returns[excess_returns < 0]

        if downside_returns.empty:
            return 999.0 if excess_returns.mean() > 0 else 0.0

        downside_std = math.sqrt((downside_returns ** 2).mean())
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return (excess_returns.mean() / downside_std) * math.sqrt(periods_per_year)

    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_drawdown_pct: float) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown %)."""
        if max_drawdown_pct <= 0:
            return 999.0 if cagr > 0 else 0.0
        return (cagr * 100.0) / max_drawdown_pct

    @staticmethod
    def calculate_ulcer_index(equity_df: pd.DataFrame) -> float:
        """Calculate Ulcer Index (downside volatility metric)."""
        if equity_df.empty or "drawdown_pct" not in equity_df.columns:
            return 0.0

        dd_pct = equity_df["drawdown_pct"]
        squared_dd = (dd_pct / 100.0) ** 2
        mean_squared = squared_dd.mean()
        return math.sqrt(mean_squared) * 100.0

    @staticmethod
    def calculate_consecutive_streaks(trade_df: pd.DataFrame) -> Tuple[int, int]:
        """Calculate max consecutive wins and max consecutive losses.

        Returns (max_consecutive_wins, max_consecutive_losses).
        """
        if trade_df.empty or "net_pnl" not in trade_df.columns:
            return (0, 0)

        results = (trade_df["net_pnl"] > 0).astype(int).tolist()

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for r in results:
            if r == 1:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return (max_wins, max_losses)
