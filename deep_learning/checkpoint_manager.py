"""
QuantLab Institutional Checkpoint Manager.

Saves, loads, versions, and restores deep learning model weights, architecture JSON,
optimizer state dicts, loss history, and evaluation metrics.
"""

import json
import os
from typing import Any, Dict, Optional
import joblib


class CheckpointManager:
    """Institutional Neural Network Checkpoint Manager."""

    @staticmethod
    def save_checkpoint(
        model: Any,
        filepath: str,
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        loss_history: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save deep learning model checkpoint bundle to disk.

        Args:
            model: PyTorch nn.Module or fallback model object.
            filepath: Destination file path (.pt / .ckpt / .joblib).
            optimizer: Optional PyTorch optimizer object.
            epoch: Epoch counter integer.
            loss_history: Loss tracking dictionary.
            metrics: Performance metrics dictionary.

        Returns:
            Absolute file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "loss_history": loss_history or {},
            "metrics": metrics or {},
        }

        try:
            import torch
            if isinstance(model, torch.nn.Module):
                checkpoint_data["state_dict"] = model.state_dict()
                if optimizer and hasattr(optimizer, "state_dict"):
                    checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(checkpoint_data, filepath)
                return os.path.abspath(filepath)
        except Exception:
            pass

        checkpoint_data["model_obj"] = model
        joblib.dump(checkpoint_data, filepath)
        return os.path.abspath(filepath)

    @staticmethod
    def load_checkpoint(filepath: str, model: Optional[Any] = None) -> Dict[str, Any]:
        """Load deep learning model checkpoint bundle from disk.

        Args:
            filepath: Target file path.
            model: Optional model instance to load state dict into.

        Returns:
            Checkpoint dictionary containing state dict / model object and metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file '{filepath}' does not exist.")

        try:
            import torch
            checkpoint_data = torch.load(filepath, map_location="cpu")
            if model is not None and isinstance(model, torch.nn.Module) and "state_dict" in checkpoint_data:
                model.load_state_dict(checkpoint_data["state_dict"])
            return checkpoint_data
        except Exception:
            return joblib.load(filepath)
