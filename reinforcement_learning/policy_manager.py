"""
QuantLab Reinforcement Learning - Policy Manager.

Manages the lifecycle of RL policy networks: save, load, clone, freeze, archive,
and export policies across PyTorch and NumPy backends.
"""

import json
import os
import pickle
from typing import Any, Dict, Optional
import numpy as np


class PolicyManager:
    """Institutional RL Policy Lifecycle Manager.

    Provides operations to save, load, clone, freeze, archive, and export
    policy networks for DQN-class, actor-critic, and policy-gradient agents.
    """

    @staticmethod
    def save(
        policy: Any,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save policy to disk using pickle (fallback) or PyTorch state dict.

        Args:
            policy: Policy network object (PyTorch Module or dict).
            output_path: Target file path (.pkl or .pt).
            metadata: Optional metadata to save alongside the policy.

        Returns:
            Absolute output file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            import torch
            if isinstance(policy, torch.nn.Module):
                bundle = {
                    "state_dict": policy.state_dict(),
                    "metadata": metadata or {},
                }
                torch.save(bundle, output_path)
                return os.path.abspath(output_path)
        except ImportError:
            pass

        # Fallback: pickle
        bundle = {"policy": policy, "metadata": metadata or {}}
        with open(output_path, "wb") as f:
            pickle.dump(bundle, f)

        return os.path.abspath(output_path)

    @staticmethod
    def load(policy_path: str, policy_obj: Optional[Any] = None) -> Any:
        """Load policy from disk.

        Args:
            policy_path: Path to the saved policy file.
            policy_obj: Optional existing policy object to load state dict into.

        Returns:
            Loaded policy or state dictionary.
        """
        try:
            import torch
            bundle = torch.load(policy_path, map_location="cpu", weights_only=False)
            if isinstance(bundle, dict) and "state_dict" in bundle:
                if policy_obj is not None and isinstance(policy_obj, torch.nn.Module):
                    policy_obj.load_state_dict(bundle["state_dict"])
                    return policy_obj
                return bundle["state_dict"]
        except Exception:
            pass

        with open(policy_path, "rb") as f:
            bundle = pickle.load(f)

        return bundle.get("policy", bundle)

    @staticmethod
    def clone(policy: Any) -> Any:
        """Create a deep copy of a policy network.

        Args:
            policy: Policy network to clone.

        Returns:
            Cloned policy object.
        """
        import copy
        return copy.deepcopy(policy)

    @staticmethod
    def freeze(policy: Any) -> Any:
        """Freeze policy network parameters (disable gradients).

        Args:
            policy: PyTorch Module policy.

        Returns:
            Frozen policy.
        """
        try:
            import torch
            if isinstance(policy, torch.nn.Module):
                for param in policy.parameters():
                    param.requires_grad = False
        except ImportError:
            pass
        return policy

    @staticmethod
    def unfreeze(policy: Any) -> Any:
        """Unfreeze policy network parameters (re-enable gradients)."""
        try:
            import torch
            if isinstance(policy, torch.nn.Module):
                for param in policy.parameters():
                    param.requires_grad = True
        except ImportError:
            pass
        return policy

    @staticmethod
    def export_json_spec(policy: Any, output_path: str) -> str:
        """Export a JSON specification of the policy architecture.

        Args:
            policy: Policy network object.
            output_path: Target JSON file path.

        Returns:
            Absolute JSON file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        spec = {
            "class": policy.__class__.__name__,
            "repr": str(policy),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        return os.path.abspath(output_path)

    @staticmethod
    def archive(policy: Any, archive_dir: str, agent_id: str) -> str:
        """Archive a policy to a versioned directory.

        Args:
            policy: Policy to archive.
            archive_dir: Root archive directory.
            agent_id: Agent identifier for subdirectory naming.

        Returns:
            Archive file path.
        """
        path = os.path.join(archive_dir, agent_id, "policy.pkl")
        return PolicyManager.save(policy, path, metadata={"status": "ARCHIVED", "agent_id": agent_id})
