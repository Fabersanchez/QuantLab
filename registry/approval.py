"""
QuantLab Institutional Governance Approval Workflow System.

Manages model and strategy approval state transitions:
DRAFT -> TRAINING -> TESTING -> VALIDATED -> APPROVED (or REJECTED / DEPRECATED / ARCHIVED).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalState(str, Enum):
    """Institutional Governance Lifecycle States."""

    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    TESTING = "TESTING"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    ARCHIVED = "ARCHIVED"
    DEPRECATED = "DEPRECATED"
    REJECTED = "REJECTED"


@dataclass
class TransitionRecord:
    """Dataclass holding individual approval state transition event."""

    old_state: ApprovalState
    new_state: ApprovalState
    approver: str
    comments: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ApprovalWorkflow:
    """Institutional Approval Workflow Engine."""

    # Valid state transition graph
    VALID_TRANSITIONS: Dict[ApprovalState, List[ApprovalState]] = {
        ApprovalState.DRAFT: [ApprovalState.TRAINING, ApprovalState.REJECTED],
        ApprovalState.TRAINING: [ApprovalState.TESTING, ApprovalState.REJECTED],
        ApprovalState.TESTING: [ApprovalState.VALIDATED, ApprovalState.REJECTED],
        ApprovalState.VALIDATED: [ApprovalState.APPROVED, ApprovalState.REJECTED],
        ApprovalState.APPROVED: [ApprovalState.DEPRECATED, ApprovalState.ARCHIVED],
        ApprovalState.DEPRECATED: [ApprovalState.ARCHIVED],
        ApprovalState.REJECTED: [ApprovalState.DRAFT, ApprovalState.ARCHIVED],
        ApprovalState.ARCHIVED: [],
    }

    def __init__(self, initial_state: ApprovalState = ApprovalState.DRAFT) -> None:
        """Initialize ApprovalWorkflow.

        Args:
            initial_state: Initial state enum.
        """
        self.current_state: ApprovalState = initial_state
        self.history: List[TransitionRecord] = []

    def transition_to(
        self, new_state: ApprovalState, approver: str = "SystemAdmin", comments: str = ""
    ) -> bool:
        """Attempt to transition record to a new approval state.

        Args:
            new_state: Target ApprovalState.
            approver: Name/ID of approver.
            comments: Reviewer comments.

        Returns:
            Boolean indicating whether transition was allowed and executed.
        """
        if new_state == self.current_state:
            return True

        valid_next = self.VALID_TRANSITIONS.get(self.current_state, [])
        if new_state not in valid_next:
            return False

        record = TransitionRecord(
            old_state=self.current_state,
            new_state=new_state,
            approver=approver,
            comments=comments,
        )
        self.history.append(record)
        self.current_state = new_state
        return True
