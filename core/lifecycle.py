"""
QuantLab System Lifecycle Manager.

Manages state transitions for the quantitative research engine and sub-modules,
enforcing strict transition rules across defined system states.
"""

from enum import Enum, auto
from typing import Dict, Set


class SystemState(Enum):
    """Enumeration of valid system states in QuantLab."""

    CREATED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class InvalidStateTransitionError(Exception):
    """Raised when an illegal system state transition is attempted."""

    pass


class LifecycleManager:
    """Controls and validates system state transitions.

    Attributes:
        current_state (SystemState): Current active lifecycle state.
    """

    # Allowed state transition map: State -> Set of allowed next States
    _ALLOWED_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
        SystemState.CREATED: {SystemState.INITIALIZING, SystemState.ERROR},
        SystemState.INITIALIZING: {SystemState.READY, SystemState.ERROR},
        SystemState.READY: {
            SystemState.RUNNING,
            SystemState.STOPPING,
            SystemState.ERROR,
        },
        SystemState.RUNNING: {
            SystemState.PAUSED,
            SystemState.STOPPING,
            SystemState.ERROR,
        },
        SystemState.PAUSED: {
            SystemState.RUNNING,
            SystemState.STOPPING,
            SystemState.ERROR,
        },
        SystemState.STOPPING: {SystemState.STOPPED, SystemState.ERROR},
        SystemState.STOPPED: {
            SystemState.INITIALIZING,
            SystemState.CREATED,
            SystemState.ERROR,
        },
        SystemState.ERROR: {SystemState.INITIALIZING, SystemState.STOPPED},
    }

    def __init__(self) -> None:
        """Initialize LifecycleManager in CREATED state."""
        self._current_state: SystemState = SystemState.CREATED

    @property
    def current_state(self) -> SystemState:
        """Return the current system state."""
        return self._current_state

    def transition_to(self, target_state: SystemState) -> SystemState:
        """Transition system to target_state if valid.

        Args:
            target_state: Desired destination SystemState.

        Returns:
            The new current SystemState.

        Raises:
            InvalidStateTransitionError: If transition is forbidden.
        """
        allowed = self._ALLOWED_TRANSITIONS.get(self._current_state, set())
        if target_state not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._current_state.name} to {target_state.name}."
            )
        self._current_state = target_state
        return self._current_state

    def is_in_state(self, state: SystemState) -> bool:
        """Check if current state matches given state."""
        return self._current_state == state
