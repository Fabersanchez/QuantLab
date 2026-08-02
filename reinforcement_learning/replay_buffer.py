"""
QuantLab Reinforcement Learning - Experience Replay Buffers.

Provides three institutional replay buffer implementations:
  1. UniformReplayBuffer - Uniform random sampling (standard DQN)
  2. PrioritizedReplayBuffer - Priority-based sampling weighted by TD-error (PER)
  3. NStepReplayBuffer - N-step return accumulation before storage
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Transition:
    """Single experience transition (s, a, r, s', done)."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class UniformReplayBuffer:
    """Standard Uniform Experience Replay Buffer for DQN-class agents."""

    def __init__(self, capacity: int = 50_000) -> None:
        """Initialize UniformReplayBuffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store a new transition in the buffer."""
        self._buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) arrays.
        """
        idx = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in idx]
        return self._unpack(batch)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check whether buffer has enough samples."""
        return len(self) >= batch_size

    @staticmethod
    def _unpack(batch: List[Transition]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer (PER) with TD-error priority sampling.

    Reference: Schaul et al. (2016) "Prioritized Experience Replay".
    """

    def __init__(
        self, capacity: int = 50_000, alpha: float = 0.6, beta: float = 0.4
    ) -> None:
        """Initialize PrioritizedReplayBuffer.

        Args:
            capacity: Maximum buffer size.
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized).
            beta: Importance sampling correction factor.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        self._buffer: List[Optional[Transition]] = [None] * capacity
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._pos: int = 0
        self._size: int = 0
        self._max_priority: float = 1.0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store transition with maximum priority."""
        t = Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        )
        self._buffer[self._pos] = t
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch proportional to TD-error priorities.

        Returns:
            (states, actions, rewards, next_states, dones, weights, indices)
        """
        probs = self._priorities[:self._size] ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(self._size, batch_size, replace=False, p=probs)
        weights = (self._size * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        batch = [self._buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = UniformReplayBuffer._unpack(batch)
        return states, actions, rewards, next_states, dones, weights.astype(np.float32), idx

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on new TD errors after gradient update."""
        for i, err in zip(indices, td_errors):
            p = float(abs(err)) + 1e-6
            self._priorities[i] = p
            self._max_priority = max(self._max_priority, p)

    def __len__(self) -> int:
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size


class NStepReplayBuffer:
    """N-Step Replay Buffer accumulating N-step bootstrapped returns.

    Stores (s_t, a_t, R_n, s_{t+n}, done) where R_n = sum(gamma^k * r_{t+k}).
    """

    def __init__(
        self, capacity: int = 50_000, n_step: int = 3, gamma: float = 0.99
    ) -> None:
        """Initialize NStepReplayBuffer.

        Args:
            capacity: Maximum buffer capacity.
            n_step: Number of steps for bootstrapped return.
            gamma: Discount factor.
        """
        self.n_step = n_step
        self.gamma = gamma
        self._n_step_buffer: deque = deque(maxlen=n_step)
        self._main_buffer = UniformReplayBuffer(capacity=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Store transition, compute N-step return when buffer is full."""
        self._n_step_buffer.append((state, action, reward, next_state, done))

        if len(self._n_step_buffer) < self.n_step:
            return

        # Compute N-step return
        state_n, action_n, _, _, _ = self._n_step_buffer[0]
        r_n = 0.0
        for k, (_, _, r, ns, d) in enumerate(self._n_step_buffer):
            r_n += (self.gamma ** k) * r
            if d:
                next_state_n = ns
                done_n = True
                break
        else:
            _, _, _, next_state_n, done_n = self._n_step_buffer[-1]

        self._main_buffer.push(state_n, action_n, r_n, next_state_n, done_n)

    def sample(self, batch_size: int) -> Tuple:
        """Delegate to main buffer uniform sampler."""
        return self._main_buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self._main_buffer)

    def is_ready(self, batch_size: int) -> bool:
        return self._main_buffer.is_ready(batch_size)
