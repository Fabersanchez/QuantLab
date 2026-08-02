"""
QuantLab Reinforcement Learning - Institutional Multi-Objective Reward Engine.

Calculates per-step rewards for RL market agents considering:
  - Realized and Unrealized PnL
  - Risk-adjusted Sharpe / Sortino components
  - Max Drawdown penalty
  - Excessive risk penalty
  - Consistency bonus
  - Capital preservation bonus
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class RewardConfig:
    """Configuration weights for the multi-objective RewardEngine."""

    pnl_weight: float = 1.0
    sharpe_weight: float = 0.5
    sortino_weight: float = 0.3
    drawdown_penalty: float = 2.0
    risk_penalty: float = 0.5
    consistency_bonus: float = 0.2
    preservation_bonus: float = 0.3
    trade_penalty: float = 0.01       # small cost per trade to discourage overtrading
    holding_penalty: float = 0.001    # tiny cost per step to incentivize decisive action


class RewardEngine:
    """Institutional Multi-Objective Reward Calculator for RL Market Agents.

    Computes a composite reward signal at every environment step combining
    profit, risk adjustment, drawdown protection, and behavioral incentives.
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        """Initialize RewardEngine.

        Args:
            config: RewardConfig with component weights. Defaults to balanced config.
        """
        self.config = config or RewardConfig()
        self._return_history: List[float] = []

    def reset(self) -> None:
        """Reset episode return history."""
        self._return_history = []

    def calculate(
        self,
        prev_equity: float,
        current_equity: float,
        peak_equity: float,
        action: int,
        position: int,
        prev_position: int,
        step: int = 0,
    ) -> float:
        """Calculate composite step reward.

        Args:
            prev_equity: Portfolio equity at previous step.
            current_equity: Portfolio equity at current step.
            peak_equity: Highest equity value reached in the episode.
            action: Integer action taken (from DiscreteAction).
            position: Current position (-1, 0, +1).
            prev_position: Previous position (-1, 0, +1).
            step: Current episode step count.

        Returns:
            Float composite reward signal.
        """
        # 1. PnL return component
        if prev_equity > 0:
            step_return = (current_equity - prev_equity) / prev_equity
        else:
            step_return = 0.0

        self._return_history.append(step_return)

        reward = self.config.pnl_weight * step_return

        # 2. Risk-adjusted Sharpe component (running Sharpe approximation)
        if len(self._return_history) >= 5:
            ret_arr = np.array(self._return_history[-50:])
            sharpe = self._sharpe(ret_arr)
            sortino = self._sortino(ret_arr)
            reward += self.config.sharpe_weight * np.clip(sharpe, -3.0, 3.0) * 0.01
            reward += self.config.sortino_weight * np.clip(sortino, -3.0, 3.0) * 0.01

        # 3. Max drawdown penalty
        if peak_equity > 0:
            dd_pct = (peak_equity - current_equity) / peak_equity
            if dd_pct > 0.05:
                reward -= self.config.drawdown_penalty * dd_pct

        # 4. Trade execution cost (discourage overtrading)
        if prev_position != position:
            reward -= self.config.trade_penalty

        # 5. Holding penalty (tiny per-step cost to encourage decisive action)
        if position == 0:
            reward -= self.config.holding_penalty

        # 6. Capital preservation bonus: reward agent for protecting equity
        if current_equity >= prev_equity and peak_equity > 0:
            if (current_equity / peak_equity) >= 0.98:
                reward += self.config.preservation_bonus * 0.001

        return float(np.clip(reward, -10.0, 10.0))

    def calculate_terminal(self, equity_history: List[float]) -> float:
        """Calculate terminal episode reward based on full equity curve.

        Args:
            equity_history: List of equity values per step.

        Returns:
            Float terminal bonus/penalty reward.
        """
        if len(equity_history) < 2:
            return 0.0

        arr = np.array(equity_history, dtype=np.float64)
        returns = np.diff(arr) / np.maximum(arr[:-1], 1.0)
        terminal = self._sharpe(returns) * 0.1

        # Total return bonus
        total_ret = (arr[-1] - arr[0]) / max(1.0, arr[0])
        terminal += self.config.pnl_weight * total_ret * 0.5

        return float(np.clip(terminal, -5.0, 5.0))

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Compute Sharpe Ratio from array of returns."""
        std = returns.std()
        if std < 1e-8:
            return 0.0
        return float((returns.mean() - risk_free) / std * np.sqrt(252))

    @staticmethod
    def _sortino(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Compute Sortino Ratio from array of returns."""
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 3.0
        dstd = downside.std()
        if dstd < 1e-8:
            return 0.0
        return float((returns.mean() - risk_free) / dstd * np.sqrt(252))
