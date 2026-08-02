"""
QuantLab Reinforcement Learning - Progressive Curriculum Learning Engine.

Manages training progression through increasingly complex market environments:
  Stage 0: Simple Trend Markets
  Stage 1: Mixed Markets (trend + range)
  Stage 2: High Volatility Markets
  Stage 3: News & Event Spike Markets
  Stage 4: Extreme Stress / Black Swan Markets
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class CurriculumStage:
    """Definition of a single curriculum learning stage."""

    stage_id: int
    name: str
    description: str
    volatility_multiplier: float = 1.0
    trend_strength: float = 0.5          # 0 = no trend, 1 = strong trend
    spike_probability: float = 0.0       # Probability of injecting a price spike per step
    spike_magnitude: float = 0.0         # Max spike magnitude as fraction of price
    max_episodes: int = 100              # Episodes to train before auto-advancing
    pass_threshold: float = 0.0         # Mean episode reward needed to advance


CURRICULUM_STAGES: List[CurriculumStage] = [
    CurriculumStage(
        stage_id=0,
        name="Simple Trend",
        description="Clean directional trending markets with low noise",
        volatility_multiplier=0.5,
        trend_strength=0.9,
        spike_probability=0.0,
        max_episodes=50,
        pass_threshold=0.1,
    ),
    CurriculumStage(
        stage_id=1,
        name="Mixed Markets",
        description="Combination of trending and ranging price regimes",
        volatility_multiplier=1.0,
        trend_strength=0.5,
        spike_probability=0.01,
        max_episodes=100,
        pass_threshold=0.05,
    ),
    CurriculumStage(
        stage_id=2,
        name="High Volatility",
        description="High-volatility regimes with rapid reversals",
        volatility_multiplier=2.5,
        trend_strength=0.3,
        spike_probability=0.03,
        spike_magnitude=0.005,
        max_episodes=100,
        pass_threshold=0.0,
    ),
    CurriculumStage(
        stage_id=3,
        name="News & Event Spikes",
        description="Markets with sudden news-driven price spikes and gap events",
        volatility_multiplier=3.0,
        trend_strength=0.2,
        spike_probability=0.08,
        spike_magnitude=0.015,
        max_episodes=100,
        pass_threshold=-0.05,
    ),
    CurriculumStage(
        stage_id=4,
        name="Extreme Stress / Black Swan",
        description="Extreme stress scenarios: flash crashes, circuit breakers, illiquid markets",
        volatility_multiplier=5.0,
        trend_strength=0.1,
        spike_probability=0.15,
        spike_magnitude=0.04,
        max_episodes=100,
        pass_threshold=-0.1,
    ),
]


class CurriculumManager:
    """Progressive Curriculum Learning Manager for RL Market Agents.

    Controls automatic stage progression based on rolling episode reward performance.
    Applies market data transformations (volatility scaling, spike injection)
    to produce synthetic training data matching each curriculum stage difficulty.
    """

    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
        window_size: int = 20,
        auto_advance: bool = True,
    ) -> None:
        """Initialize CurriculumManager.

        Args:
            stages: List of CurriculumStage definitions.
            window_size: Rolling episode window for advancement evaluation.
            auto_advance: Whether to automatically advance stages.
        """
        self.stages = stages or CURRICULUM_STAGES
        self.window_size = window_size
        self.auto_advance = auto_advance

        self._current_stage_idx: int = 0
        self._episode_rewards: List[float] = []
        self._stage_episode_count: int = 0
        self._completed_stages: List[int] = []

    @property
    def current_stage(self) -> CurriculumStage:
        """Return current active curriculum stage."""
        return self.stages[self._current_stage_idx]

    @property
    def stage_idx(self) -> int:
        """Return current stage index."""
        return self._current_stage_idx

    @property
    def is_complete(self) -> bool:
        """Return True if all curriculum stages have been completed."""
        return self._current_stage_idx >= len(self.stages) - 1

    def log_episode_reward(self, reward: float) -> bool:
        """Log completed episode reward and evaluate stage advancement.

        Args:
            reward: Total reward from completed episode.

        Returns:
            True if stage was advanced, False otherwise.
        """
        self._episode_rewards.append(reward)
        self._stage_episode_count += 1
        return self._try_advance()

    def _try_advance(self) -> bool:
        """Check and execute stage advancement if criteria are met."""
        if not self.auto_advance or self.is_complete:
            return False

        stage = self.current_stage

        # Check episodes requirement
        if self._stage_episode_count < stage.max_episodes:
            return False

        # Check performance requirement
        window = self._episode_rewards[-self.window_size:]
        mean_reward = float(np.mean(window)) if window else -np.inf

        if mean_reward >= stage.pass_threshold:
            self._advance()
            return True

        return False

    def force_advance(self) -> bool:
        """Force advance to next stage regardless of performance."""
        if self.is_complete:
            return False
        self._advance()
        return True

    def _advance(self) -> None:
        """Move to next curriculum stage."""
        self._completed_stages.append(self._current_stage_idx)
        self._current_stage_idx = min(self._current_stage_idx + 1, len(self.stages) - 1)
        self._stage_episode_count = 0
        self._episode_rewards = []

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply current curriculum stage transformations to market DataFrame.

        Scales volatility, injects trend drift, and adds random price spikes
        to simulate the current stage's market difficulty.

        Args:
            df: Source OHLCV DataFrame.

        Returns:
            Transformed DataFrame matching current stage characteristics.
        """
        stage = self.current_stage
        df_out = df.copy()

        if "close" not in df_out.columns:
            return df_out

        close = df_out["close"].values.copy()
        returns = np.diff(close) / np.maximum(close[:-1], 1.0)

        # Scale volatility
        returns = returns * stage.volatility_multiplier

        # Add trend drift
        drift = stage.trend_strength * 0.0001
        returns = returns + drift

        # Inject spikes
        if stage.spike_probability > 0 and stage.spike_magnitude > 0:
            spike_mask = np.random.random(len(returns)) < stage.spike_probability
            spikes = np.random.uniform(-stage.spike_magnitude, stage.spike_magnitude, len(returns))
            returns[spike_mask] += spikes[spike_mask]

        # Reconstruct price series
        new_close = np.zeros(len(close))
        new_close[0] = close[0]
        for i in range(1, len(close)):
            new_close[i] = new_close[i - 1] * (1 + returns[i - 1])

        scale = close / np.maximum(new_close, 1e-8)
        df_out["close"] = new_close
        if "high" in df_out.columns:
            df_out["high"] = df_out["high"] / scale
        if "low" in df_out.columns:
            df_out["low"] = df_out["low"] / scale
        if "open" in df_out.columns:
            df_out["open"] = df_out["open"] / scale

        return df_out

    def summary(self) -> Dict[str, Any]:
        """Return curriculum progress summary dictionary."""
        return {
            "current_stage": self.current_stage.name,
            "stage_idx": self._current_stage_idx,
            "total_stages": len(self.stages),
            "is_complete": self.is_complete,
            "stage_episodes": self._stage_episode_count,
            "completed_stages": self._completed_stages,
        }
