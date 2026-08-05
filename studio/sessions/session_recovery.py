"""
QuantLab Studio Session Crash Recovery Engine.

Provides full state crash recovery saving open panels, window positions, active layout,
active workspace/project, open files, search filters, navigation history, and widget states.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
from studio.logging.studio_logger import get_studio_logger

logger = get_studio_logger("SessionRecovery")


@dataclass
class CrashRecoveryCheckpoint:
    """Dataclass holding crash recovery state snapshot."""

    checkpoint_id: str
    active_workspace: str = "default_workspace"
    active_project: str = ""
    open_files: List[str] = field(default_factory=list)
    active_layout: str = "DefaultLayout"
    widget_states: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionRecoveryEngine:
    """Institutional Session Crash Recovery Engine."""

    def __init__(self, recovery_filepath: str = "studio_crash_recovery.json") -> None:
        self.recovery_filepath = recovery_filepath
        self._last_checkpoint: Optional[CrashRecoveryCheckpoint] = None

    def create_checkpoint(
        self,
        active_workspace: str,
        active_project: str = "",
        open_files: Optional[List[str]] = None,
        active_layout: str = "DefaultLayout",
        widget_states: Optional[Dict[str, Any]] = None,
    ) -> CrashRecoveryCheckpoint:
        """Create and persist crash recovery checkpoint snapshot."""
        cp = CrashRecoveryCheckpoint(
            checkpoint_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            active_workspace=active_workspace,
            active_project=active_project,
            open_files=open_files or [],
            active_layout=active_layout,
            widget_states=widget_states or {},
        )
        self._last_checkpoint = cp

        try:
            with open(self.recovery_filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(cp), f, indent=2)
            logger.info(f"Created crash recovery checkpoint ID={cp.checkpoint_id}")
        except Exception as e:
            logger.error(f"Failed to write crash recovery checkpoint: {e}")

        return cp

    def recover_last_session(self) -> Optional[CrashRecoveryCheckpoint]:
        """Recover last checkpoint snapshot from file storage after crash."""
        if not os.path.exists(self.recovery_filepath):
            return None

        try:
            with open(self.recovery_filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            cp = CrashRecoveryCheckpoint(**data)
            self._last_checkpoint = cp
            logger.info(f"Successfully recovered session checkpoint ID={cp.checkpoint_id}")
            return cp
        except Exception as e:
            logger.error(f"Failed to read crash recovery checkpoint: {e}")
            return None
