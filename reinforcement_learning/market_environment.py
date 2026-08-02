"""
QuantLab Reinforcement Learning - Financial Market Environment.

Implements a Gymnasium-compatible market simulation environment that represents
OHLCV price series with spread, commission, slippage, and order latency.
Agents interact with this environment by taking discrete trading actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from reinforcement_learning.action_space import ActionSpace, DiscreteAction
from reinforcement_learning.environment import BaseEnvironment, SpaceSpec, StepResult
from reinforcement_learning.reward_engine import RewardConfig, RewardEngine
from reinforcement_learning.state_builder import PortfolioState, StateBuilder


@dataclass
class MarketConfig:
    """Configuration for MarketEnvironment simulation parameters."""

    initial_equity: float = 10_000.0
    spread_pct: float = 0.0002         # 2 pips spread (as fraction of price)
    commission_pct: float = 0.0001     # 0.01% commission per trade
    slippage_pct: float = 0.0001       # 0.01% slippage
    latency_steps: int = 0             # Execution latency (in bars)
    max_position_size: float = 1.0     # Max fraction of equity to allocate
    stop_loss_pct: float = 0.02        # 2% hard stop loss
    take_profit_pct: float = 0.04      # 4% take profit
    lookback: int = 20                 # State observation lookback window
    feature_cols: List[str] = field(default_factory=list)
    max_steps: int = 500               # Max steps per episode
    allow_shorting: bool = True


class MarketEnvironment(BaseEnvironment):
    """Gymnasium-compatible Financial Market Simulation Environment.

    Represents the market as a sequential OHLCV time series where an RL agent
    receives market state observations and takes discrete trading actions.
    Execution costs (spread, commission, slippage) are fully modeled.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[MarketConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize MarketEnvironment.

        Args:
            df: OHLCV DataFrame (must contain 'open', 'high', 'low', 'close', 'volume').
            config: MarketConfig simulation parameters.
            reward_config: RewardConfig for the reward engine.
            seed: Optional random seed.
        """
        super().__init__(seed=seed)
        self._df = df.reset_index(drop=True)
        self.config = config or MarketConfig()
        self._action_space = ActionSpace()
        self._state_builder = StateBuilder(
            lookback=self.config.lookback,
            feature_cols=self.config.feature_cols,
        )
        self._reward_engine = RewardEngine(reward_config)

        # Portfolio state
        self._portfolio = PortfolioState(
            equity=self.config.initial_equity,
            cash=self.config.initial_equity,
        )

        self._current_idx: int = self.config.lookback
        self._equity_history: List[float] = []
        self._trade_log: List[Dict[str, Any]] = []

    @property
    def observation_space(self) -> SpaceSpec:
        """Return flat observation space specification."""
        return SpaceSpec(
            shape=(self._state_builder.obs_dim,),
            dtype=np.dtype("float32"),
            low=np.full(self._state_builder.obs_dim, -np.inf, dtype=np.float32),
            high=np.full(self._state_builder.obs_dim, np.inf, dtype=np.float32),
        )

    @property
    def action_space(self) -> SpaceSpec:
        """Return discrete action space specification."""
        return self._action_space.spec

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset market environment to initial state.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        if seed is not None:
            self.seed(seed)

        self._current_idx = self.config.lookback
        self._episode_count += 1
        self._step_count = 0
        self._done = False

        self._portfolio = PortfolioState(
            equity=self.config.initial_equity,
            cash=self.config.initial_equity,
            peak_equity=self.config.initial_equity,
        )
        self._equity_history = [self.config.initial_equity]
        self._trade_log = []
        self._reward_engine.reset()

        obs = self._build_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> StepResult:
        """Execute one simulation step with the given action.

        Args:
            action: Integer action from DiscreteAction.

        Returns:
            StepResult with observation, reward, done, truncated, info.
        """
        if self._done:
            obs = self._build_obs()
            return StepResult(observation=obs, reward=0.0, done=True, truncated=False)

        prev_equity = self._portfolio.equity
        prev_position = self._portfolio.position

        # Execute trade action
        self._execute_action(action)

        # Update unrealized PnL
        current_price = self._get_price()
        self._update_unrealized_pnl(current_price)

        # Update equity
        self._portfolio.equity = self._portfolio.cash + self._portfolio.unrealized_pnl
        self._portfolio.equity = max(0.01, self._portfolio.equity)

        # Update peak equity & drawdown
        if self._portfolio.equity > self._portfolio.peak_equity:
            self._portfolio.peak_equity = self._portfolio.equity
        if self._portfolio.peak_equity > 0:
            self._portfolio.drawdown_pct = (
                (self._portfolio.peak_equity - self._portfolio.equity) / self._portfolio.peak_equity
            )

        self._equity_history.append(self._portfolio.equity)

        # Calculate reward
        reward = self._reward_engine.calculate(
            prev_equity=prev_equity,
            current_equity=self._portfolio.equity,
            peak_equity=self._portfolio.peak_equity,
            action=action,
            position=self._portfolio.position,
            prev_position=prev_position,
            step=self._step_count,
        )

        # Advance step
        self._step_count += 1
        self._current_idx += 1

        # Check episode termination
        truncated = self._step_count >= self.config.max_steps
        done = (
            self._current_idx >= len(self._df) - 1
            or self._portfolio.equity < self.config.initial_equity * 0.5  # 50% ruin stop
            or truncated
        )

        if done:
            reward += self._reward_engine.calculate_terminal(self._equity_history)

        self._done = done
        obs = self._build_obs()
        info = self._get_info()
        info.update({
            "equity": self._portfolio.equity,
            "position": self._portfolio.position,
            "drawdown_pct": self._portfolio.drawdown_pct,
            "n_trades": self._portfolio.n_trades,
            "action_label": self._action_space.to_label(action),
        })

        return StepResult(observation=obs, reward=reward, done=done, truncated=truncated, info=info)

    def render(self, mode: str = "human") -> Optional[str]:
        """Render current environment state as human-readable string."""
        price = self._get_price()
        output = (
            f"[MarketEnv] Step={self._step_count} | Bar={self._current_idx} | "
            f"Price={price:.4f} | Equity={self._portfolio.equity:.2f} | "
            f"Position={self._portfolio.position} | DD={self._portfolio.drawdown_pct*100:.1f}%"
        )
        if mode == "human":
            print(output)
        return output

    def close(self) -> None:
        """Release environment resources."""
        self._trade_log.clear()
        self._equity_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_price(self, col: str = "close") -> float:
        """Get current bar closing price."""
        idx = min(self._current_idx, len(self._df) - 1)
        return float(self._df.at[idx, col])

    def _get_spread_cost(self, price: float) -> float:
        return price * self.config.spread_pct

    def _get_commission(self, price: float, size: float) -> float:
        return price * size * self.config.commission_pct

    def _get_slippage(self, price: float) -> float:
        slippage = price * self.config.slippage_pct
        return float(self._rng.uniform(0.0, slippage))

    def _execute_action(self, action: int) -> None:
        """Execute trade action and update portfolio state."""
        price = self._get_price()

        if action == DiscreteAction.HOLD:
            return

        elif action == DiscreteAction.BUY:
            if self._portfolio.position == 0 or (self.config.allow_shorting and self._portfolio.position == -1):
                self._open_position(price, direction=1)

        elif action == DiscreteAction.SELL:
            if self.config.allow_shorting and (self._portfolio.position == 0 or self._portfolio.position == 1):
                self._open_position(price, direction=-1)

        elif action == DiscreteAction.CLOSE:
            if self._portfolio.position != 0:
                self._close_position(price)

        elif action == DiscreteAction.PARTIAL_CLOSE:
            if self._portfolio.position != 0:
                # Close half of current position
                self._portfolio.position_size *= 0.5
                pnl = (price - self._portfolio.entry_price) * self._portfolio.position * self._portfolio.position_size
                self._portfolio.cash += pnl
                self._portfolio.realized_pnl += pnl

        elif action in (DiscreteAction.MODIFY_SL, DiscreteAction.MODIFY_TP):
            # Conceptual modifications - no direct equity effect in simulation
            pass

    def _open_position(self, price: float, direction: int) -> None:
        """Open a new market position."""
        if self._portfolio.position != 0:
            self._close_position(price)

        exec_price = price + self._get_slippage(price) + self._get_spread_cost(price) * direction
        size = (self._portfolio.cash * self.config.max_position_size) / max(exec_price, 0.01)
        commission = self._get_commission(exec_price, size)
        self._portfolio.cash -= commission

        self._portfolio.position = direction
        self._portfolio.position_size = size
        self._portfolio.entry_price = exec_price
        self._portfolio.unrealized_pnl = 0.0
        self._portfolio.n_trades += 1

        self._trade_log.append({
            "step": self._step_count,
            "action": "OPEN",
            "direction": direction,
            "price": exec_price,
            "size": size,
            "commission": commission,
        })

    def _close_position(self, price: float) -> None:
        """Close the current open position."""
        if self._portfolio.position == 0:
            return

        exec_price = price - self._get_slippage(price) * self._portfolio.position
        pnl = (exec_price - self._portfolio.entry_price) * self._portfolio.position * self._portfolio.position_size
        commission = self._get_commission(exec_price, self._portfolio.position_size)

        self._portfolio.cash += pnl - commission
        self._portfolio.realized_pnl += pnl
        self._portfolio.unrealized_pnl = 0.0
        self._portfolio.position = 0
        self._portfolio.position_size = 0.0
        self._portfolio.entry_price = 0.0

        self._trade_log.append({
            "step": self._step_count,
            "action": "CLOSE",
            "price": exec_price,
            "pnl": pnl,
            "commission": commission,
        })

    def _update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized PnL for open position."""
        if self._portfolio.position != 0 and self._portfolio.entry_price > 0:
            self._portfolio.unrealized_pnl = (
                (current_price - self._portfolio.entry_price)
                * self._portfolio.position
                * self._portfolio.position_size
            )

    def _build_obs(self) -> np.ndarray:
        """Build the observation vector for the current state."""
        idx = min(self._current_idx, len(self._df) - 1)
        return self._state_builder.build(
            df=self._df,
            current_idx=idx,
            portfolio=self._portfolio,
        )

    @property
    def trade_log(self) -> list:
        """Return list of executed trade records."""
        return self._trade_log.copy()

    @property
    def equity_history(self) -> List[float]:
        """Return list of equity values per episode step."""
        return self._equity_history.copy()
