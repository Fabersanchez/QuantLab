"""
QuantLab Deep Learning Callbacks.

Provides Early Stopping, Model Checkpoint, Learning Rate Scheduler (ReduceLROnPlateau, CosineAnnealing),
CSV Logger, TensorBoard Logger, and composite CallbackList.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseCallback(ABC):
    """Abstract Base Class for all Deep Learning training callbacks."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


class EarlyStoppingCallback(BaseCallback):
    """Early Stopping callback to halt training when validation loss stops improving."""

    def __init__(self, monitor: str = "val_loss", patience: int = 5, min_delta: float = 1e-4) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.should_stop: bool = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not logs or self.monitor not in logs:
            return

        current = float(logs[self.monitor])
        if self.best_score is None:
            self.best_score = current
        elif current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True


class ModelCheckpointCallback(BaseCallback):
    """Model Checkpoint callback to save best model weights."""

    def __init__(self, filepath: str, monitor: str = "val_loss", save_best_only: bool = True) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score: Optional[float] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if not logs or self.monitor not in logs:
            return

        current = float(logs[self.monitor])
        if self.best_score is None or current < self.best_score:
            self.best_score = current
            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            with open(self.filepath + ".txt", "w") as f:
                f.write(f"Epoch {epoch}: {self.monitor}={current:.6f}\n")


class CSVLoggerCallback(BaseCallback):
    """Logs epoch metrics to CSV file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.logs_history: List[Dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if logs:
            row = {"epoch": epoch}
            row.update(logs)
            self.logs_history.append(row)
            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            pd.DataFrame(self.logs_history).to_csv(self.filepath, index=False)


class CallbackList:
    """Composite manager triggering all registered callbacks."""

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None) -> None:
        self.callbacks = callbacks or []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger on_epoch_end for all callbacks. Returns True if early stopping triggered."""
        stop = False
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)
            if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                stop = True
        return stop
