"""
QuantLab Reinforcement Learning - MLOps RL Agent Registry.

Registers, versions, tracks lifecycle status transitions, and queries
RL agents including their algorithm family, hyperparameters, and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


# Valid RL agent lifecycle status values
VALID_STATUSES = ("EXPERIMENTAL", "STAGING", "PRODUCTION", "ARCHIVED")

# Supported RL algorithm identifiers
SUPPORTED_ALGORITHMS = (
    "DQN", "DOUBLE_DQN", "DUELING_DQN", "RAINBOW",
    "PPO", "A2C", "A3C",
    "SAC", "TD3", "DDPG",
)


@dataclass
class RLAgentRecord:
    """Dataclass encapsulating a registered RL agent record."""

    agent_id: str
    name: str
    algorithm: str
    version: int
    hyperparameters: Dict[str, Any]
    dataset_info: Dict[str, Any]
    metrics: Dict[str, float]
    author: str
    status: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checkpoint_path: Optional[str] = None
    notes: str = ""


class RLAgentRegistry:
    """Institutional MLOps Reinforcement Learning Agent Registry.

    Registers, versions, and manages RL agents with lifecycle state transitions:
    EXPERIMENTAL -> STAGING -> PRODUCTION -> ARCHIVED.
    """

    def __init__(self) -> None:
        """Initialize RLAgentRegistry."""
        self._agents: Dict[str, RLAgentRecord] = {}
        self._version_counter: Dict[str, int] = {}

    def register(
        self,
        name: str,
        algorithm: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        author: str = "QuantLabRL",
        status: str = "EXPERIMENTAL",
        checkpoint_path: Optional[str] = None,
        notes: str = "",
    ) -> RLAgentRecord:
        """Register a new RL agent version.

        Args:
            name: Agent name identifier.
            algorithm: RL algorithm family (e.g., 'DQN', 'PPO', 'SAC').
            hyperparameters: Dictionary of hyperparameter key-value pairs.
            dataset_info: Dictionary of dataset/environment metadata.
            metrics: Performance metrics dictionary.
            author: Author or team identifier.
            status: Lifecycle status (EXPERIMENTAL, STAGING, PRODUCTION, ARCHIVED).
            checkpoint_path: Optional path to agent checkpoint directory.
            notes: Free-text notes.

        Returns:
            RLAgentRecord instance.
        """
        algo = algorithm.upper().strip()

        version = self._version_counter.get(name, 0) + 1
        self._version_counter[name] = version

        agent_id = f"RL-{algo}-{name.upper()}-v{version}-{uuid.uuid4().hex[:6]}"

        record = RLAgentRecord(
            agent_id=agent_id,
            name=name,
            algorithm=algo,
            version=version,
            hyperparameters=hyperparameters or {},
            dataset_info=dataset_info or {},
            metrics=metrics or {},
            author=author,
            status=status.upper(),
            checkpoint_path=checkpoint_path,
            notes=notes,
        )

        self._agents[agent_id] = record
        return record

    def get(self, agent_id: str) -> Optional[RLAgentRecord]:
        """Fetch agent record by ID."""
        return self._agents.get(agent_id)

    def update_status(self, agent_id: str, new_status: str) -> RLAgentRecord:
        """Transition agent lifecycle status.

        Args:
            agent_id: Agent identifier.
            new_status: Target status string.

        Returns:
            Updated RLAgentRecord.
        """
        record = self.get(agent_id)
        if not record:
            raise KeyError(f"Agent '{agent_id}' not found in registry.")

        st = new_status.upper().strip()
        if st not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of: {VALID_STATUSES}")

        record.status = st
        return record

    def update_metrics(self, agent_id: str, metrics: Dict[str, float]) -> RLAgentRecord:
        """Update agent performance metrics."""
        record = self.get(agent_id)
        if not record:
            raise KeyError(f"Agent '{agent_id}' not found in registry.")
        record.metrics.update(metrics)
        return record

    def list_agents(
        self,
        status_filter: Optional[str] = None,
        algorithm_filter: Optional[str] = None,
    ) -> List[RLAgentRecord]:
        """List registered agents with optional filters."""
        records = list(self._agents.values())
        if status_filter:
            records = [r for r in records if r.status == status_filter.upper()]
        if algorithm_filter:
            records = [r for r in records if r.algorithm == algorithm_filter.upper()]
        return records

    def get_best_agent(
        self, metric: str = "mean_reward", higher_is_better: bool = True
    ) -> Optional[RLAgentRecord]:
        """Find best performing registered agent by metric."""
        candidates = [r for r in self._agents.values() if metric in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics[metric]) if higher_is_better else min(
            candidates, key=lambda r: r.metrics[metric]
        )
