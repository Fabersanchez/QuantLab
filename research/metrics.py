"""
QuantLab Institutional Centralized Metrics Engine.

Implements decoupled independent metric calculation objects adhering to Clean Architecture.
Provides complete financial, risk-adjusted, drawdown, trade execution, exposure, and cost impact
analytical metrics for quantitative scientific research.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class BaseMetric(ABC):
    """Abstract Base Class for all decoupled research metrics."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize BaseMetric.

        Args:
            name: Metric unique name identifier.
            description: Detailed metric description.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> Union[float, int, Dict[str, float]]:
        """Compute the metric value.

        Args:
            trades_df: DataFrame containing trade logs (columns: pnl, pnl_pct, duration, commission, slippage, etc.).
            equity_series: Series of portfolio equity values over time.
            returns_series: Series of percentage returns over time.
            initial_capital: Initial account capital.
            risk_free_rate: Annualized risk-free interest rate.
            total_bars: Total duration in candles/bars.

        Returns:
            Calculated metric value or metric breakdown dictionary.
        """
        pass


class ProfitFactorMetric(BaseMetric):
    """Decoupled metric calculating Profit Factor (Gross Profit / Gross Loss)."""

    def __init__(self) -> None:
        super().__init__("profit_factor", "Ratio of gross profits to absolute gross losses")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        pnls = trades_df["pnl"].values
        gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
        gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.0
        if gross_loss == 0.0:
            return float(gross_profit) if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)


class NetProfitMetric(BaseMetric):
    """Decoupled metric calculating Net Profit."""

    def __init__(self) -> None:
        super().__init__("net_profit", "Total net profit achieved across all trades")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if equity_series is not None and not equity_series.empty:
            return float(equity_series.iloc[-1] - equity_series.iloc[0])
        if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
            return float(trades_df["pnl"].sum())
        return 0.0


class GrossProfitMetric(BaseMetric):
    """Decoupled metric calculating Gross Profit."""

    def __init__(self) -> None:
        super().__init__("gross_profit", "Sum of all positive winning trade PnLs")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        pnls = trades_df["pnl"].values
        return float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0


class GrossLossMetric(BaseMetric):
    """Decoupled metric calculating Gross Loss."""

    def __init__(self) -> None:
        super().__init__("gross_loss", "Sum of all negative losing trade PnLs")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        pnls = trades_df["pnl"].values
        return float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.0


class AverageTradeMetric(BaseMetric):
    """Decoupled metric calculating Average Trade PnL."""

    def __init__(self) -> None:
        super().__init__("average_trade", "Mean PnL outcome per completed trade")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        return float(trades_df["pnl"].mean())


class PayoffRatioMetric(BaseMetric):
    """Decoupled metric calculating Payoff Ratio (Avg Win / Avg Loss)."""

    def __init__(self) -> None:
        super().__init__("payoff_ratio", "Ratio of average win to absolute average loss")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
        if avg_loss == 0.0:
            return float(avg_win) if avg_win > 0 else 0.0
        return float(avg_win / avg_loss)


class ExpectancyMetric(BaseMetric):
    """Decoupled metric calculating Expectancy in monetary value."""

    def __init__(self) -> None:
        super().__init__("expectancy", "Expected PnL per trade based on win/loss probabilities")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        win_rate = len(wins) / total_trades
        loss_rate = len(losses) / total_trades
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(abs(losses.mean())) if not losses.empty else 0.0
        return float((win_rate * avg_win) - (loss_rate * avg_loss))


class SharpeRatioMetric(BaseMetric):
    """Decoupled metric calculating Sharpe Ratio."""

    def __init__(self) -> None:
        super().__init__("sharpe_ratio", "Annualized Sharpe Ratio measuring risk-adjusted excess returns")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if returns_series is None or returns_series.empty:
            if equity_series is not None and len(equity_series) > 1:
                returns_series = equity_series.pct_change().dropna()
            else:
                return 0.0
        mean_ret = float(returns_series.mean())
        std_ret = float(returns_series.std())
        if std_ret == 0.0 or np.isnan(std_ret):
            return 0.0
        rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
        sharpe = ((mean_ret - rf_daily) / std_ret) * np.sqrt(252)
        return float(sharpe) if not np.isnan(sharpe) else 0.0


class SortinoRatioMetric(BaseMetric):
    """Decoupled metric calculating Sortino Ratio."""

    def __init__(self) -> None:
        super().__init__("sortino_ratio", "Annualized Sortino Ratio measuring downside risk-adjusted returns")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if returns_series is None or returns_series.empty:
            if equity_series is not None and len(equity_series) > 1:
                returns_series = equity_series.pct_change().dropna()
            else:
                return 0.0
        rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
        downside = returns_series[returns_series < rf_daily] - rf_daily
        if len(downside) == 0:
            return 0.0
        downside_std = float(np.sqrt(np.mean(downside**2)))
        if downside_std == 0.0 or np.isnan(downside_std):
            return 0.0
        mean_ret = float(returns_series.mean())
        sortino = ((mean_ret - rf_daily) / downside_std) * np.sqrt(252)
        return float(sortino) if not np.isnan(sortino) else 0.0


class CalmarRatioMetric(BaseMetric):
    """Decoupled metric calculating Calmar Ratio."""

    def __init__(self) -> None:
        super().__init__("calmar_ratio", "Ratio of Annualized Return to Maximum Drawdown")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        ann_return = AnnualReturnMetric().compute(
            trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital, total_bars=total_bars
        )
        max_dd = MaxDrawdownMetric().compute(equity_series=equity_series)
        if max_dd == 0.0:
            return float(ann_return) if ann_return > 0 else 0.0
        return float(ann_return / max_dd)


class RecoveryFactorMetric(BaseMetric):
    """Decoupled metric calculating Recovery Factor (Net Profit / Max Drawdown monetary value)."""

    def __init__(self) -> None:
        super().__init__("recovery_factor", "Ratio of Net Profit to monetary Maximum Drawdown")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        net_profit = NetProfitMetric().compute(
            trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital
        )
        if equity_series is None or len(equity_series) < 2:
            return 0.0
        peak = equity_series.cummax()
        dd_dollars = peak - equity_series
        max_dd_dollars = float(dd_dollars.max())
        if max_dd_dollars == 0.0:
            return float(net_profit) if net_profit > 0 else 0.0
        return float(net_profit / max_dd_dollars)


class UlcerIndexMetric(BaseMetric):
    """Decoupled metric calculating Ulcer Index."""

    def __init__(self) -> None:
        super().__init__("ulcer_index", "Square root of mean squared percentage drawdowns")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if equity_series is None or len(equity_series) < 2:
            return 0.0
        peak = equity_series.cummax()
        pct_dd = ((equity_series - peak) / peak) * 100.0
        squared_dd = pct_dd**2
        return float(np.sqrt(np.mean(squared_dd)))


class SQNMetric(BaseMetric):
    """Decoupled metric calculating System Quality Number (Van Tharp SQN)."""

    def __init__(self) -> None:
        super().__init__("sqn", "System Quality Number measuring trade expectancy consistency")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        pnls = trades_df["pnl"].values
        n = len(pnls)
        if n < 2:
            return 0.0
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1))
        if std_pnl == 0.0:
            return 0.0
        sqn = (avg_pnl / std_pnl) * np.sqrt(n)
        return float(sqn)


class KellyCriterionMetric(BaseMetric):
    """Decoupled metric calculating Optimal Kelly Criterion fraction."""

    def __init__(self) -> None:
        super().__init__("kelly_criterion", "Optimal theoretical leverage/position size fraction")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        if wins.empty or losses.empty:
            return 0.0
        w = len(wins) / total_trades
        r = float(wins.mean()) / float(abs(losses.mean()))
        if r == 0.0:
            return 0.0
        kelly = w - ((1.0 - w) / r)
        return float(max(0.0, kelly))


class MaxDrawdownMetric(BaseMetric):
    """Decoupled metric calculating Maximum Drawdown Percentage."""

    def __init__(self) -> None:
        super().__init__("max_drawdown", "Maximum percentage decline from peak to trough")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if equity_series is None or len(equity_series) < 2:
            return 0.0
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        return float(abs(drawdown.min())) * 100.0


class DrawdownDurationMetric(BaseMetric):
    """Decoupled metric calculating Maximum and Average Drawdown duration in bars."""

    def __init__(self) -> None:
        super().__init__("drawdown_duration", "Maximum bars spent in drawdown recovery state")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if equity_series is None or len(equity_series) < 2:
            return 0.0
        peak = equity_series.cummax()
        is_underwater = equity_series < peak

        current_dur = 0
        max_dur = 0
        for underwater in is_underwater:
            if underwater:
                current_dur += 1
                if current_dur > max_dur:
                    max_dur = current_dur
            else:
                current_dur = 0
        return float(max_dur)


class WinRateMetric(BaseMetric):
    """Decoupled metric calculating Win Rate Percentage."""

    def __init__(self) -> None:
        super().__init__("win_rate", "Percentage of profitable trades")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        wins = np.sum(trades_df["pnl"] > 0)
        return float((wins / len(trades_df)) * 100.0)


class LossRateMetric(BaseMetric):
    """Decoupled metric calculating Loss Rate Percentage."""

    def __init__(self) -> None:
        super().__init__("loss_rate", "Percentage of unprofitable trades")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        losses = np.sum(trades_df["pnl"] < 0)
        return float((losses / len(trades_df)) * 100.0)


class AverageWinMetric(BaseMetric):
    """Decoupled metric calculating Average Winning Trade PnL."""

    def __init__(self) -> None:
        super().__init__("average_win", "Mean monetary return of winning trades")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        wins = trades_df[trades_df["pnl"] > 0]["pnl"]
        return float(wins.mean()) if not wins.empty else 0.0


class AverageLossMetric(BaseMetric):
    """Decoupled metric calculating Average Losing Trade PnL."""

    def __init__(self) -> None:
        super().__init__("average_loss", "Mean monetary loss of losing trades")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        losses = trades_df[trades_df["pnl"] < 0]["pnl"]
        return float(abs(losses.mean())) if not losses.empty else 0.0


class LargestWinMetric(BaseMetric):
    """Decoupled metric calculating Largest Winning Trade."""

    def __init__(self) -> None:
        super().__init__("largest_win", "Maximum monetary gain in a single trade")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        return float(trades_df["pnl"].max())


class LargestLossMetric(BaseMetric):
    """Decoupled metric calculating Largest Losing Trade."""

    def __init__(self) -> None:
        super().__init__("largest_loss", "Maximum monetary loss in a single trade")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        return float(abs(trades_df["pnl"].min()))


class ConsecutiveWinsMetric(BaseMetric):
    """Decoupled metric calculating Maximum Consecutive Winning Trades."""

    def __init__(self) -> None:
        super().__init__("consecutive_wins", "Maximum length of consecutive winning streak")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> int:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0
        pnls = trades_df["pnl"].values
        current_streak = 0
        max_streak = 0
        for val in pnls:
            if val > 0:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0
        return int(max_streak)


class ConsecutiveLossesMetric(BaseMetric):
    """Decoupled metric calculating Maximum Consecutive Losing Trades."""

    def __init__(self) -> None:
        super().__init__("consecutive_losses", "Maximum length of consecutive losing streak")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> int:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0
        pnls = trades_df["pnl"].values
        current_streak = 0
        max_streak = 0
        for val in pnls:
            if val < 0:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 0
        return int(max_streak)


class VolatilityMetric(BaseMetric):
    """Decoupled metric calculating Annualized Volatility Percentage."""

    def __init__(self) -> None:
        super().__init__("volatility", "Annualized standard deviation of returns percentage")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if returns_series is None or returns_series.empty:
            if equity_series is not None and len(equity_series) > 1:
                returns_series = equity_series.pct_change().dropna()
            else:
                return 0.0
        std = float(returns_series.std())
        if np.isnan(std):
            return 0.0
        return float(std * np.sqrt(252) * 100.0)


class ReturnMetric(BaseMetric):
    """Decoupled metric calculating Total Percentage Return."""

    def __init__(self) -> None:
        super().__init__("total_return", "Overall portfolio cumulative percentage return")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        net_profit = NetProfitMetric().compute(
            trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital
        )
        if initial_capital <= 0.0:
            return 0.0
        return float((net_profit / initial_capital) * 100.0)


class AnnualReturnMetric(BaseMetric):
    """Decoupled metric calculating Compound Annual Growth Rate (CAGR)."""

    def __init__(self) -> None:
        super().__init__("cagr", "Compound Annual Growth Rate percentage")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        final_equity = (
            float(equity_series.iloc[-1])
            if equity_series is not None and not equity_series.empty
            else initial_capital
            + NetProfitMetric().compute(
                trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital
            )
        )
        if initial_capital <= 0.0 or final_equity <= 0.0:
            return 0.0
        bars = total_bars if total_bars > 0 else (len(equity_series) if equity_series is not None else 252)
        years = max(bars / 252.0, 1.0 / 252.0)
        cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0
        return float(cagr) if not np.isnan(cagr) else 0.0


class CommissionImpactMetric(BaseMetric):
    """Decoupled metric calculating Commission Impact in dollars and percentage of returns."""

    def __init__(self) -> None:
        super().__init__("commission_impact", "Total commissions incurred")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "commission" not in trades_df.columns:
            return 0.0
        return float(trades_df["commission"].sum())


class SlippageImpactMetric(BaseMetric):
    """Decoupled metric calculating Slippage Impact in dollars."""

    def __init__(self) -> None:
        super().__init__("slippage_impact", "Total slippage costs incurred")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "slippage" not in trades_df.columns:
            return 0.0
        return float(trades_df["slippage"].sum())


class RiskRewardMetric(BaseMetric):
    """Decoupled metric calculating Risk Reward Ratio."""

    def __init__(self) -> None:
        super().__init__("risk_reward", "Ratio of average reward to average risk setup")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        return PayoffRatioMetric().compute(trades_df=trades_df)


class ExpectancyScoreMetric(BaseMetric):
    """Decoupled metric calculating Expectancy Score."""

    def __init__(self) -> None:
        super().__init__("expectancy_score", "Normalized Expectancy Score per unit of risk")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "pnl" not in trades_df.columns:
            return 0.0
        exp = ExpectancyMetric().compute(trades_df=trades_df)
        avg_loss = AverageLossMetric().compute(trades_df=trades_df)
        if avg_loss == 0.0:
            return float(exp) if exp > 0 else 0.0
        return float(exp / avg_loss)


class TradeDurationMetric(BaseMetric):
    """Decoupled metric calculating Average Trade Duration in bars."""

    def __init__(self) -> None:
        super().__init__("trade_duration", "Average bars held per position")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "duration" not in trades_df.columns:
            return 0.0
        return float(trades_df["duration"].mean())


class ExposureMetric(BaseMetric):
    """Decoupled metric calculating Exposure / Time in Market percentage."""

    def __init__(self) -> None:
        super().__init__("exposure", "Percentage of time portfolio holds active positions")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if trades_df is None or trades_df.empty or "duration" not in trades_df.columns or total_bars <= 0:
            return 0.0
        total_trade_bars = trades_df["duration"].sum()
        return float(min(100.0, (total_trade_bars / total_bars) * 100.0))


class MonthlyReturnMetric(BaseMetric):
    """Decoupled metric calculating Average Monthly Return Percentage."""

    def __init__(self) -> None:
        super().__init__("monthly_return", "Average monthly return percentage")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        cagr = AnnualReturnMetric().compute(
            trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital, total_bars=total_bars
        )
        return float(((1.0 + cagr / 100.0) ** (1.0 / 12.0) - 1.0) * 100.0)


class DailyReturnMetric(BaseMetric):
    """Decoupled metric calculating Average Daily Return Percentage."""

    def __init__(self) -> None:
        super().__init__("daily_return", "Average daily return percentage")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        if returns_series is not None and not returns_series.empty:
            return float(returns_series.mean() * 100.0)
        cagr = AnnualReturnMetric().compute(
            trades_df=trades_df, equity_series=equity_series, initial_capital=initial_capital, total_bars=total_bars
        )
        return float(((1.0 + cagr / 100.0) ** (1.0 / 252.0) - 1.0) * 100.0)


class MarketExposureMetric(BaseMetric):
    """Decoupled metric calculating Market Exposure percentage."""

    def __init__(self) -> None:
        super().__init__("market_exposure", "Percentage of market exposure during active session")

    def compute(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> float:
        return ExposureMetric().compute(trades_df=trades_df, total_bars=total_bars)


class MetricsCalculator:
    """Master Orchestrator for QuantLab centralized metrics evaluation."""

    def __init__(self) -> None:
        """Initialize MetricsCalculator registering all decoupled independent metrics."""
        self.metrics: Dict[str, BaseMetric] = {
            "profit_factor": ProfitFactorMetric(),
            "net_profit": NetProfitMetric(),
            "gross_profit": GrossProfitMetric(),
            "gross_loss": GrossLossMetric(),
            "average_trade": AverageTradeMetric(),
            "payoff_ratio": PayoffRatioMetric(),
            "expectancy": ExpectancyMetric(),
            "sharpe_ratio": SharpeRatioMetric(),
            "sortino_ratio": SortinoRatioMetric(),
            "calmar_ratio": CalmarRatioMetric(),
            "recovery_factor": RecoveryFactorMetric(),
            "ulcer_index": UlcerIndexMetric(),
            "sqn": SQNMetric(),
            "kelly_criterion": KellyCriterionMetric(),
            "max_drawdown": MaxDrawdownMetric(),
            "drawdown_duration": DrawdownDurationMetric(),
            "win_rate": WinRateMetric(),
            "loss_rate": LossRateMetric(),
            "average_win": AverageWinMetric(),
            "average_loss": AverageLossMetric(),
            "largest_win": LargestWinMetric(),
            "largest_loss": LargestLossMetric(),
            "consecutive_wins": ConsecutiveWinsMetric(),
            "consecutive_losses": ConsecutiveLossesMetric(),
            "volatility": VolatilityMetric(),
            "total_return": ReturnMetric(),
            "cagr": AnnualReturnMetric(),
            "monthly_return": MonthlyReturnMetric(),
            "daily_return": DailyReturnMetric(),
            "commission_impact": CommissionImpactMetric(),
            "slippage_impact": SlippageImpactMetric(),
            "risk_reward": RiskRewardMetric(),
            "expectancy_score": ExpectancyScoreMetric(),
            "trade_duration": TradeDurationMetric(),
            "exposure": ExposureMetric(),
            "market_exposure": MarketExposureMetric(),
        }

    def compute_all(
        self,
        trades_df: Optional[pd.DataFrame] = None,
        equity_series: Optional[pd.Series] = None,
        returns_series: Optional[pd.Series] = None,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        total_bars: int = 0,
    ) -> Dict[str, Any]:
        """Compute all registered metric values.

        Returns:
            Dictionary mapping metric names to their calculated analytical values.
        """
        results: Dict[str, Any] = {}
        for key, metric in self.metrics.items():
            try:
                results[key] = metric.compute(
                    trades_df=trades_df,
                    equity_series=equity_series,
                    returns_series=returns_series,
                    initial_capital=initial_capital,
                    risk_free_rate=risk_free_rate,
                    total_bars=total_bars,
                )
            except Exception:
                results[key] = 0.0
        return results
