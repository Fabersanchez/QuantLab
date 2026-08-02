"""
QuantLab Reinforcement Learning - Discrete & Continuous Action Space.

Defines financial trading action space for RL agents:
HOLD, BUY, SELL, CLOSE, PARTIAL_CLOSE, MODIFY_SL, MODIFY_TP.
"""

from enum import IntEnum
from typing import Dict, List, Optional
import numpy as np

from reinforcement_learning.environment import SpaceSpec


class DiscreteAction(IntEnum):
    """Enumeration of all valid discrete trading actions for RL agents."""

    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3
    PARTIAL_CLOSE = 4
    MODIFY_SL = 5
    MODIFY_TP = 6


# Human-readable labels for each action
ACTION_LABELS: Dict[int, str] = {
    DiscreteAction.HOLD: "HOLD",
    DiscreteAction.BUY: "BUY",
    DiscreteAction.SELL: "SELL",
    DiscreteAction.CLOSE: "CLOSE",
    DiscreteAction.PARTIAL_CLOSE: "PARTIAL_CLOSE",
    DiscreteAction.MODIFY_SL: "MODIFY_SL",
    DiscreteAction.MODIFY_TP: "MODIFY_TP",
}

N_DISCRETE_ACTIONS: int = len(DiscreteAction)


class ActionSpace:
    """Institutional RL Discrete Trading Action Space Manager.

    Manages the mapping from integer action indices to DiscreteAction enum members
    and provides validation, sampling, and masking utilities for market agents.
    """

    def __init__(self, n_actions: int = N_DISCRETE_ACTIONS) -> None:
        """Initialize ActionSpace.

        Args:
            n_actions: Number of discrete actions. Defaults to 7 (all trading actions).
        """
        self.n_actions: int = n_actions
        self._rng = np.random.default_rng()

    @property
    def spec(self) -> SpaceSpec:
        """Return action space specification (discrete)."""
        return SpaceSpec(shape=(1,), dtype=np.dtype("int32"), n=self.n_actions)

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        """Sample a random valid action from the action space.

        Args:
            mask: Optional boolean array of length n_actions. True = valid action.

        Returns:
            Random valid action integer.
        """
        if mask is not None:
            valid = np.where(mask)[0]
            if len(valid) == 0:
                return int(DiscreteAction.HOLD)
            return int(self._rng.choice(valid))
        return int(self._rng.integers(0, self.n_actions))

    def is_valid(self, action: int) -> bool:
        """Check if action integer is valid within the space."""
        return 0 <= action < self.n_actions

    def to_label(self, action: int) -> str:
        """Convert integer action to human-readable label."""
        return ACTION_LABELS.get(action, f"ACTION_{action}")

    def all_actions(self) -> List[DiscreteAction]:
        """Return list of all available DiscreteAction members."""
        return [DiscreteAction(i) for i in range(self.n_actions)]

    def __repr__(self) -> str:
        return f"ActionSpace(n_actions={self.n_actions}, actions={[self.to_label(i) for i in range(self.n_actions)]})"
