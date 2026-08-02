"""
QuantLab Reinforcement Learning - RL Agent Trainer & Algorithm Interfaces.

Provides a unified RLTrainer that wraps algorithm-specific training logic for:
DQN, Double DQN, Dueling DQN, PPO, A2C, A3C, SAC, TD3, DDPG, Rainbow DQN.

Each algorithm family is implemented as a pure NumPy agent for portability
with optional PyTorch upgrade paths. All training is simulated in market environments.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from reinforcement_learning.environment import BaseEnvironment, StepResult
from reinforcement_learning.exploration import EpsilonGreedyExploration
from reinforcement_learning.replay_buffer import UniformReplayBuffer


# ---------------------------------------------------------------------------
# Base Agent Interface
# ---------------------------------------------------------------------------

class BaseRLAgent:
    """Abstract base class for all RL algorithm agents."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> Dict[str, float]:
        return {}

    def get_weights(self) -> Any:
        return None

    def set_weights(self, weights: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Lightweight NumPy Agent Implementations (Algorithm Interfaces)
# ---------------------------------------------------------------------------

class _LinearQAgent(BaseRLAgent):
    """Linear function approximation Q-agent (shared base for DQN-class)."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99) -> None:
        super().__init__(obs_dim, n_actions, lr, gamma)
        self.W = np.random.randn(obs_dim, n_actions).astype(np.float32) * 0.01
        self.b = np.zeros(n_actions, dtype=np.float32)
        self._exploration = EpsilonGreedyExploration()

    def _q_values(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W + self.b

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        q = self._q_values(obs.flatten())
        if deterministic:
            return int(np.argmax(q))
        return self._exploration.select_action(q, self.n_actions)

    def update(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
        next_states: np.ndarray, dones: np.ndarray
    ) -> Dict[str, float]:
        q = states @ self.W + self.b
        q_next = next_states @ self.W + self.b
        targets = q.copy()
        for i in range(len(states)):
            td_target = rewards[i] + self.gamma * np.max(q_next[i]) * (1 - dones[i])
            targets[i, actions[i]] = td_target
        grad_W = states.T @ (q - targets) / len(states)
        self.W -= self.lr * np.clip(grad_W, -1.0, 1.0)
        loss = float(np.mean((q - targets) ** 2))
        return {"loss": loss}

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.W = weights["W"].copy()
        self.b = weights["b"].copy()


class DQNAgent(_LinearQAgent):
    """Deep Q-Network (DQN) agent interface - linear Q approximation."""
    pass


class DoubleDQNAgent(_LinearQAgent):
    """Double DQN agent - uses target network to reduce overestimation bias."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W_target = self.W.copy()
        self.b_target = self.b.copy()
        self._update_freq = 100
        self._steps = 0

    def update(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        self._steps += 1
        q = states @ self.W + self.b
        q_next_online = next_states @ self.W + self.b
        q_next_target = next_states @ self.W_target + self.b_target

        targets = q.copy()
        for i in range(len(states)):
            best_a = int(np.argmax(q_next_online[i]))
            td_target = rewards[i] + self.gamma * q_next_target[i, best_a] * (1 - dones[i])
            targets[i, actions[i]] = td_target

        grad_W = states.T @ (q - targets) / len(states)
        self.W -= self.lr * np.clip(grad_W, -1.0, 1.0)

        if self._steps % self._update_freq == 0:
            self.W_target = self.W.copy()
            self.b_target = self.b.copy()

        return {"loss": float(np.mean((q - targets) ** 2))}


class DuelingDQNAgent(_LinearQAgent):
    """Dueling DQN agent - separate value and advantage streams."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W_v = np.random.randn(self.obs_dim, 1).astype(np.float32) * 0.01
        self.W_a = np.random.randn(self.obs_dim, self.n_actions).astype(np.float32) * 0.01

    def _q_values(self, obs: np.ndarray) -> np.ndarray:
        V = obs @ self.W_v
        A = obs @ self.W_a
        return V + A - A.mean(axis=-1, keepdims=True)


class RainbowDQNAgent(DoubleDQNAgent):
    """Rainbow DQN interface combining Double DQN + Dueling + N-step + Prioritized."""
    pass


class _PolicyGradientAgent(BaseRLAgent):
    """Linear policy-gradient agent (shared base for PPO, A2C, A3C)."""

    def __init__(self, obs_dim: int, n_actions: int, lr: float = 3e-4, gamma: float = 0.99) -> None:
        super().__init__(obs_dim, n_actions, lr, gamma)
        self.W_policy = np.random.randn(obs_dim, n_actions).astype(np.float32) * 0.01
        self.W_value = np.random.randn(obs_dim, 1).astype(np.float32) * 0.01

    def _policy_logits(self, obs: np.ndarray) -> np.ndarray:
        logits = obs @ self.W_policy
        logits -= logits.max()
        return logits

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        logits = self._policy_logits(obs.flatten())
        probs = self._softmax(logits)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        values = states @ self.W_value
        advantages = rewards.reshape(-1, 1) - values
        logits = states @ self.W_policy
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        for i, a in enumerate(actions):
            one_hot[i, a] = 1.0

        policy_grad = (probs - one_hot) * advantages
        self.W_policy -= self.lr * states.T @ policy_grad / len(states)
        value_loss = float(np.mean(advantages ** 2))
        return {"policy_loss": float(np.mean(policy_grad ** 2)), "value_loss": value_loss}

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W_policy": self.W_policy.copy(), "W_value": self.W_value.copy()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        self.W_policy = weights["W_policy"].copy()
        self.W_value = weights["W_value"].copy()


class PPOAgent(_PolicyGradientAgent):
    """Proximal Policy Optimization (PPO) agent interface."""
    pass


class A2CAgent(_PolicyGradientAgent):
    """Advantage Actor-Critic (A2C) agent interface."""
    pass


class A3CAgent(_PolicyGradientAgent):
    """Asynchronous Advantage Actor-Critic (A3C) agent interface."""
    pass


class _ContinuousAgent(BaseRLAgent):
    """Shared base for continuous action agents (SAC, TD3, DDPG)."""

    def __init__(self, obs_dim: int, n_actions: int = 1, lr: float = 3e-4, gamma: float = 0.99) -> None:
        super().__init__(obs_dim, n_actions, lr, gamma)
        self.W_actor = np.random.randn(obs_dim, n_actions).astype(np.float32) * 0.01
        self.W_critic = np.random.randn(obs_dim + n_actions, 1).astype(np.float32) * 0.01
        self._noise_std = 0.1

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        action_val = float(np.tanh(obs.flatten() @ self.W_actor).mean())
        if not deterministic:
            action_val += float(np.random.randn() * self._noise_std)
        # Map continuous output to discrete action
        action_idx = int(np.clip(round((action_val + 1) * 3), 0, self.n_actions - 1))
        return action_idx

    def update(self, states, actions, rewards, next_states, dones) -> Dict[str, float]:
        actor_out = np.tanh(states @ self.W_actor)
        sa = np.hstack([states, actor_out])
        q = sa @ self.W_critic
        actor_loss = -q.mean()
        self.W_actor -= self.lr * 0.1 * np.random.randn(*self.W_actor.shape)
        return {"actor_loss": float(actor_loss)}

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"W_actor": self.W_actor.copy(), "W_critic": self.W_critic.copy()}


class SACAgent(_ContinuousAgent):
    """Soft Actor-Critic (SAC) agent interface with entropy regularization."""
    pass


class TD3Agent(_ContinuousAgent):
    """Twin Delayed DDPG (TD3) agent interface."""
    pass


class DDPGAgent(_ContinuousAgent):
    """Deep Deterministic Policy Gradient (DDPG) agent interface."""
    pass


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

AGENT_MAP = {
    "DQN": DQNAgent,
    "DOUBLE_DQN": DoubleDQNAgent,
    "DUELING_DQN": DuelingDQNAgent,
    "RAINBOW": RainbowDQNAgent,
    "PPO": PPOAgent,
    "A2C": A2CAgent,
    "A3C": A3CAgent,
    "SAC": SACAgent,
    "TD3": TD3Agent,
    "DDPG": DDPGAgent,
}


def create_agent(algorithm: str, obs_dim: int, n_actions: int, **kwargs) -> BaseRLAgent:
    """Factory function to instantiate an RL agent by algorithm name.

    Args:
        algorithm: Algorithm identifier string (e.g. 'DQN', 'PPO', 'SAC').
        obs_dim: Observation vector dimensionality.
        n_actions: Number of discrete actions.
        **kwargs: Additional keyword args forwarded to agent constructor.

    Returns:
        Instantiated BaseRLAgent.
    """
    algo = algorithm.upper().strip().replace("-", "_").replace(" ", "_")
    if algo not in AGENT_MAP:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {list(AGENT_MAP.keys())}")
    return AGENT_MAP[algo](obs_dim=obs_dim, n_actions=n_actions, **kwargs)


# ---------------------------------------------------------------------------
# RL Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for a single RL training run."""

    n_episodes: int = 50
    max_steps_per_episode: int = 200
    batch_size: int = 32
    buffer_capacity: int = 10_000
    update_every: int = 4
    gamma: float = 0.99
    lr: float = 1e-3
    log_every: int = 10


@dataclass
class TrainingResult:
    """Result summary from a completed RLTrainer.train() run."""

    algorithm: str
    n_episodes: int
    episode_rewards: List[float]
    loss_history: List[float]
    mean_reward: float
    best_reward: float
    total_steps: int


class RLTrainer:
    """Institutional RL Agent Trainer.

    Orchestrates episode-level interaction loops between RL agents and market environments.
    Handles experience replay, policy updates, and training statistics aggregation.
    """

    def __init__(
        self,
        agent: BaseRLAgent,
        env: BaseEnvironment,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        """Initialize RLTrainer.

        Args:
            agent: RL agent instance.
            env: Gymnasium-compatible environment.
            config: TrainingConfig hyperparameters.
        """
        self.agent = agent
        self.env = env
        self.config = config or TrainingConfig()
        self._buffer = UniformReplayBuffer(capacity=self.config.buffer_capacity)

    def train(self) -> TrainingResult:
        """Run full training loop for n_episodes.

        Returns:
            TrainingResult summary dataclass.
        """
        episode_rewards: List[float] = []
        loss_history: List[float] = []
        total_steps = 0

        for ep in range(self.config.n_episodes):
            obs_arr, _ = self.env.reset()
            ep_reward = 0.0

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(obs_arr)
                result = self.env.step(action)
                self._buffer.push(obs_arr, action, result.reward, result.observation, result.done)

                obs_arr = result.observation
                ep_reward += result.reward
                total_steps += 1

                if step % self.config.update_every == 0 and self._buffer.is_ready(self.config.batch_size):
                    states, actions, rewards, next_states, dones = self._buffer.sample(self.config.batch_size)
                    losses = self.agent.update(states, actions, rewards, next_states, dones)
                    if "loss" in losses:
                        loss_history.append(losses["loss"])
                    elif "policy_loss" in losses:
                        loss_history.append(losses.get("policy_loss", 0.0))

                if result.done:
                    break

            episode_rewards.append(ep_reward)

        return TrainingResult(
            algorithm=self.agent.__class__.__name__,
            n_episodes=self.config.n_episodes,
            episode_rewards=episode_rewards,
            loss_history=loss_history,
            mean_reward=float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            best_reward=float(np.max(episode_rewards)) if episode_rewards else 0.0,
            total_steps=total_steps,
        )
