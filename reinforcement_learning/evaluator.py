"""
QuantLab Reinforcement Learning - RL Agent Evaluator.

Evaluates RL agent performance over multiple market environment episodes,
computing institutional trading metrics: Total Reward, Sharpe Ratio,
Sortino Ratio, Max Drawdown, Win Rate, Profit Factor, and Stability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from reinforcement_learning.environment import BaseEnvironment
from reinforcement_learning.trainer import BaseRLAgent


@dataclass
class RLEvaluationReport:
    """Structured RL evaluation results across multiple evaluation episodes."""

    algorithm: str
    n_eval_episodes: int
    episode_rewards: List[float]
    equity_histories: List[List[float]]
    metrics: Dict[str, float]
    action_counts: Dict[int, int] = field(default_factory=dict)


class RLEvaluator:
    """Institutional RL Agent Evaluator.

    Runs evaluation episodes in a market environment using the trained policy
    in deterministic (greedy) mode and computes comprehensive performance metrics.
    """

    @staticmethod
    def evaluate(
        agent: BaseRLAgent,
        env: BaseEnvironment,
        n_episodes: int = 10,
        max_steps: int = 500,
    ) -> RLEvaluationReport:
        """Evaluate RL agent over n_episodes in deterministic mode.

        Args:
            agent: Trained RL agent.
            env: Market environment to evaluate against.
            n_episodes: Number of evaluation episodes.
            max_steps: Maximum steps per episode.

        Returns:
            RLEvaluationReport with metrics.
        """
        episode_rewards: List[float] = []
        equity_histories: List[List[float]] = []
        action_counts: Dict[int, int] = {}

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0

            for _ in range(max_steps):
                action = agent.select_action(obs, deterministic=True)
                action_counts[action] = action_counts.get(action, 0) + 1
                result = env.step(action)
                obs = result.observation
                ep_reward += result.reward
                if result.done:
                    break

            episode_rewards.append(ep_reward)

            # Try to collect equity history from environment if available
            eq_hist = getattr(env, "equity_history", [])
            equity_histories.append(list(eq_hist))

        metrics = RLEvaluator._compute_metrics(episode_rewards, equity_histories)

        return RLEvaluationReport(
            algorithm=agent.__class__.__name__,
            n_eval_episodes=n_episodes,
            episode_rewards=episode_rewards,
            equity_histories=equity_histories,
            metrics=metrics,
            action_counts=action_counts,
        )

    @staticmethod
    def _compute_metrics(
        episode_rewards: List[float],
        equity_histories: List[List[float]],
    ) -> Dict[str, float]:
        """Compute comprehensive RL performance metrics."""
        rewards_arr = np.array(episode_rewards, dtype=np.float64)

        mean_reward = float(rewards_arr.mean()) if len(rewards_arr) > 0 else 0.0
        std_reward = float(rewards_arr.std()) if len(rewards_arr) > 0 else 0.0
        best_reward = float(rewards_arr.max()) if len(rewards_arr) > 0 else 0.0
        worst_reward = float(rewards_arr.min()) if len(rewards_arr) > 0 else 0.0

        # Stability: inverse of coefficient of variation
        stability = float(1.0 / (std_reward / max(abs(mean_reward), 1e-8) + 1e-6))
        stability = min(stability, 10.0)

        # Sharpe of episode rewards
        sharpe = 0.0
        if std_reward > 1e-8:
            sharpe = float((mean_reward / std_reward) * np.sqrt(len(episode_rewards)))

        # Sortino
        neg_rewards = rewards_arr[rewards_arr < 0]
        sortino = 0.0
        if len(neg_rewards) > 0 and neg_rewards.std() > 1e-8:
            sortino = float(mean_reward / neg_rewards.std() * np.sqrt(len(episode_rewards)))

        # Max Drawdown across equity histories
        max_dd = 0.0
        total_return = 0.0
        if equity_histories:
            for eq_hist in equity_histories:
                if len(eq_hist) >= 2:
                    arr = np.array(eq_hist)
                    peak = np.maximum.accumulate(arr)
                    dd_series = (peak - arr) / np.maximum(peak, 1.0)
                    max_dd = max(max_dd, float(dd_series.max()))
                    total_return += (arr[-1] - arr[0]) / max(arr[0], 1.0)
            total_return /= len(equity_histories)

        # Win rate (episodes with positive reward)
        win_rate = float((rewards_arr > 0).mean()) if len(rewards_arr) > 0 else 0.0

        # Profit factor
        gross_profit = float(rewards_arr[rewards_arr > 0].sum()) if (rewards_arr > 0).any() else 0.0
        gross_loss = float(abs(rewards_arr[rewards_arr < 0].sum())) if (rewards_arr < 0).any() else 1e-6
        profit_factor = gross_profit / max(gross_loss, 1e-6)

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "best_reward": best_reward,
            "worst_reward": worst_reward,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "stability": stability,
            "total_return": total_return,
        }
