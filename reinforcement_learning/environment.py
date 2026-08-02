"""
QuantLab Reinforcement Learning - Base Environment Interface.

Defines a Gymnasium-compatible abstract base environment that all RL environments
in QuantLab must implement: reset(), step(), render(), close(), seed().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class StepResult:
    """Structured result from environment.step() call."""

    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpaceSpec:
    """Specification of an observation or action space."""

    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    n: Optional[int] = None  # for discrete spaces


class BaseEnvironment(ABC):
    """Abstract Gymnasium-compatible base environment for QuantLab RL agents.

    All market simulation environments must inherit from BaseEnvironment
    and implement the required interface.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize BaseEnvironment."""
        self._seed: Optional[int] = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._step_count: int = 0
        self._episode_count: int = 0
        self._done: bool = False

    @property
    @abstractmethod
    def observation_space(self) -> SpaceSpec:
        """Return observation space specification."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> SpaceSpec:
        """Return action space specification."""
        ...

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional environment-specific options.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        ...

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Execute one environment step with the given action.

        Args:
            action: Integer action from action space.

        Returns:
            StepResult with observation, reward, done, truncated, info.
        """
        ...

    def render(self, mode: str = "human") -> Optional[str]:
        """Render current environment state.

        Args:
            mode: Rendering mode ('human', 'rgb_array', 'ansi').

        Returns:
            Rendered output string or None.
        """
        return None

    def close(self) -> None:
        """Release environment resources cleanly."""
        pass

    def seed(self, seed: Optional[int] = None) -> int:
        """Set random seed for environment reproducibility.

        Args:
            seed: Random seed integer.

        Returns:
            The seed used.
        """
        self._seed = seed if seed is not None else np.random.randint(0, 2**31 - 1)
        self._rng = np.random.default_rng(self._seed)
        return self._seed

    def _get_info(self) -> Dict[str, Any]:
        """Return base diagnostic info dictionary."""
        return {
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "done": self._done,
        }
