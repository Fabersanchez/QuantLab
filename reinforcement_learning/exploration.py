"""
QuantLab Reinforcement Learning - Exploration Strategy Library.

Provides five institutional exploration strategies for RL trading agents:
  1. EpsilonGreedyExploration - Classic decaying epsilon-greedy
  2. SoftmaxExploration - Boltzmann temperature-based action distribution
  3. UCBExploration - Upper Confidence Bound action selection
  4. EntropyExploration - Maximum entropy action selection
  5. NoisyNetworksExploration - Parametric noise injection (NoisyNet interface)
"""

from typing import Optional
import numpy as np


class EpsilonGreedyExploration:
    """Decaying Epsilon-Greedy exploration strategy.

    Action selection:
        - With probability epsilon: random action (explore)
        - With probability 1-epsilon: greedy action (exploit)
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_steps: int = 10_000,
    ) -> None:
        """Initialize EpsilonGreedyExploration."""
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self._step: int = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon value."""
        return max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self._step / self.decay_steps,
        )

    def select_action(self, q_values: np.ndarray, n_actions: int) -> int:
        """Select action using epsilon-greedy policy."""
        self._step += 1
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, n_actions))
        return int(np.argmax(q_values))

    def reset(self) -> None:
        """Reset step counter."""
        self._step = 0


class SoftmaxExploration:
    """Softmax (Boltzmann) Exploration - action probability proportional to Q-value magnitude.

    Controls temperature parameter tau:
        - High tau: near-uniform random exploration
        - Low tau: near-greedy exploitation
    """

    def __init__(
        self,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        decay_steps: int = 10_000,
    ) -> None:
        """Initialize SoftmaxExploration."""
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.decay_steps = decay_steps
        self._step: int = 0

    @property
    def tau(self) -> float:
        """Current temperature value."""
        return max(
            self.tau_end,
            self.tau_start - (self.tau_start - self.tau_end) * self._step / self.decay_steps,
        )

    def select_action(self, q_values: np.ndarray, n_actions: int) -> int:
        """Select action using Boltzmann softmax distribution."""
        self._step += 1
        tau = max(self.tau, 1e-6)
        scaled = q_values / tau
        scaled -= scaled.max()  # numerical stability
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(np.random.choice(len(q_values), p=probs))


class UCBExploration:
    """Upper Confidence Bound (UCB1) Exploration for Q-value action selection.

    Selects action that maximizes: Q(s,a) + c * sqrt(log(t) / N(a))
    """

    def __init__(self, c: float = 1.0, n_actions: int = 7) -> None:
        """Initialize UCBExploration.

        Args:
            c: Exploration constant scaling factor.
            n_actions: Number of discrete actions.
        """
        self.c = c
        self.n_actions = n_actions
        self._t: int = 0
        self._action_counts: np.ndarray = np.zeros(n_actions, dtype=np.float64)

    def select_action(self, q_values: np.ndarray, n_actions: Optional[int] = None) -> int:
        """Select action using UCB1 criterion."""
        self._t += 1
        n = n_actions or self.n_actions

        # Initialize: select unvisited actions first
        unvisited = np.where(self._action_counts[:n] == 0)[0]
        if len(unvisited) > 0:
            action = int(unvisited[0])
        else:
            ucb = q_values[:n] + self.c * np.sqrt(np.log(self._t) / self._action_counts[:n])
            action = int(np.argmax(ucb))

        self._action_counts[action] += 1
        return action

    def reset(self) -> None:
        """Reset action counts."""
        self._t = 0
        self._action_counts = np.zeros(self.n_actions, dtype=np.float64)


class EntropyExploration:
    """Maximum Entropy Exploration - adds entropy regularization to action selection.

    Selects actions from distribution proportional to exp(Q/temperature),
    encouraging agents to maintain diverse action distributions.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        """Initialize EntropyExploration."""
        self.temperature = max(1e-6, temperature)

    def select_action(self, q_values: np.ndarray, n_actions: int) -> int:
        """Sample action from entropy-regularized distribution."""
        scaled = q_values[:n_actions] / self.temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(np.random.choice(n_actions, p=probs))

    def entropy(self, q_values: np.ndarray, n_actions: int) -> float:
        """Compute current action distribution entropy."""
        scaled = q_values[:n_actions] / self.temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return float(-np.sum(probs * np.log(probs + 1e-8)))


class NoisyNetworksExploration:
    """Noisy Networks Exploration Interface (NoisyNet).

    Provides parametric Gaussian noise injection as a drop-in replacement
    for epsilon-greedy. Actual weight noise is injected in the PyTorch model layers.
    This class provides the interface for resetting and controlling noise levels.
    """

    def __init__(self, sigma: float = 0.5) -> None:
        """Initialize NoisyNetworksExploration.

        Args:
            sigma: Initial noise standard deviation for NoisyLinear layers.
        """
        self.sigma = sigma
        self._step: int = 0

    def select_action(self, q_values: np.ndarray, n_actions: int) -> int:
        """Greedy action selection (noise is embedded in the network weights)."""
        self._step += 1
        return int(np.argmax(q_values[:n_actions]))

    def reset_noise(self) -> None:
        """Signal to network layers to resample noise parameters."""
        self._step += 1

    @property
    def step(self) -> int:
        """Return current step count."""
        return self._step
