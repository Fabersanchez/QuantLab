"""
QuantLab Reinforcement Learning - Checkpoint Manager.

Saves and loads complete RL agent checkpoints including policy weights,
value function weights, optimizer state dicts, reward history, curriculum stage,
and episode statistics.
"""

import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RLCheckpoint:
    """Dataclass representing a complete RL agent checkpoint bundle."""

    agent_id: str
    algorithm: str
    episode: int
    step: int
    policy_weights: Any
    value_weights: Optional[Any]
    optimizer_state: Optional[Any]
    reward_history: List[float]
    curriculum_stage: int
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    saved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RLCheckpointManager:
    """Institutional RL Agent Checkpoint Manager.

    Provides save/load operations for RL training checkpoints,
    best-checkpoint tracking, and checkpoint history listing.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/rl") -> None:
        """Initialize RLCheckpointManager.

        Args:
            checkpoint_dir: Root directory for storing checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        agent_id: str,
        algorithm: str,
        episode: int,
        step: int,
        policy_weights: Any,
        value_weights: Optional[Any] = None,
        optimizer_state: Optional[Any] = None,
        reward_history: Optional[List[float]] = None,
        curriculum_stage: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tag: str = "latest",
    ) -> str:
        """Save RL agent checkpoint to disk.

        Args:
            agent_id: Unique agent identifier.
            episode: Current training episode number.
            step: Current global step count.
            policy_weights: Policy network weights (state_dict or array).
            value_weights: Optional value network weights.
            optimizer_state: Optional optimizer state dict.
            reward_history: Episode reward history list.
            curriculum_stage: Current curriculum stage index.
            metrics: Performance metrics snapshot.
            hyperparameters: Hyperparameter dictionary.
            tag: Checkpoint tag ('latest', 'best', or episode number).

        Returns:
            Absolute path to saved checkpoint file.
        """
        ckpt = RLCheckpoint(
            agent_id=agent_id,
            algorithm=algorithm,
            episode=episode,
            step=step,
            policy_weights=policy_weights,
            value_weights=value_weights,
            optimizer_state=optimizer_state,
            reward_history=reward_history or [],
            curriculum_stage=curriculum_stage,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
        )

        agent_dir = os.path.join(self.checkpoint_dir, agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        path = os.path.join(agent_dir, f"{tag}.ckpt")

        try:
            import torch
            if hasattr(policy_weights, "state_dict"):
                ckpt.policy_weights = policy_weights.state_dict()
            if value_weights is not None and hasattr(value_weights, "state_dict"):
                ckpt.value_weights = value_weights.state_dict()
            torch.save(ckpt, path)
        except Exception:
            with open(path, "wb") as f:
                pickle.dump(ckpt, f)

        # Save metadata sidecar JSON
        meta_path = os.path.join(agent_dir, f"{tag}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "agent_id": agent_id,
                "algorithm": algorithm,
                "episode": episode,
                "step": step,
                "curriculum_stage": curriculum_stage,
                "metrics": metrics or {},
                "saved_at": ckpt.saved_at,
            }, f, indent=2)

        return os.path.abspath(path)

    def load(self, agent_id: str, tag: str = "latest") -> Optional[RLCheckpoint]:
        """Load RL agent checkpoint from disk.

        Args:
            agent_id: Agent identifier.
            tag: Checkpoint tag to load.

        Returns:
            RLCheckpoint object or None if not found.
        """
        path = os.path.join(self.checkpoint_dir, agent_id, f"{tag}.ckpt")
        if not os.path.exists(path):
            return None

        try:
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            pass

        with open(path, "rb") as f:
            return pickle.load(f)

    def list_checkpoints(self, agent_id: str) -> List[str]:
        """List all available checkpoint tags for an agent."""
        agent_dir = os.path.join(self.checkpoint_dir, agent_id)
        if not os.path.exists(agent_dir):
            return []
        return [
            f.replace(".ckpt", "")
            for f in os.listdir(agent_dir)
            if f.endswith(".ckpt")
        ]

    def delete(self, agent_id: str, tag: str = "latest") -> bool:
        """Delete a checkpoint by tag."""
        path = os.path.join(self.checkpoint_dir, agent_id, f"{tag}.ckpt")
        if os.path.exists(path):
            os.remove(path)
            meta = path.replace(".ckpt", "_meta.json")
            if os.path.exists(meta):
                os.remove(meta)
            return True
        return False
