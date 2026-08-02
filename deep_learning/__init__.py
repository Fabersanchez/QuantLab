"""
QuantLab Institutional Deep Learning Research Lab Package.

Provides complete MLOps and neural network infrastructure tailored for financial time series:
Sliding window sequence builder, financial dataset builder, institutional DataLoader (batching, cache, memmap),
sequence pre-processing, time series data augmentation (noise, time warping, mixup),
neural network model factory (MLP, 1D CNN, LSTM, BiLSTM, GRU, Transformer, Temporal Transformer, TCN, AutoEncoder, VAE, Hybrid CNN-LSTM),
training callbacks (Early Stopping, Model Checkpoint, LR Scheduler, CSV Logger), Checkpoint Manager,
DLTrainer, DLEvaluator, DLPredictor inference engine, DLModelRegistry, DLExperimentTracker,
DLExportManager (ONNX, TorchScript, HDF5), DLVisualizer (loss curves SVG, attention maps SVG),
DLReportGenerator (HTML, Markdown, PDF, JSON, CSV), and master DeepLearningEngine.
"""

from deep_learning.sequence_builder import SequenceBuilder
from deep_learning.dataset_builder import TimeSeriesDataset, DatasetBuilder
from deep_learning.dataloader import DLDataLoader
from deep_learning.preprocessing import DLPreprocessor
from deep_learning.augmentation import TimeSeriesAugmenter
from deep_learning.model_factory import PyTorchBaseModel, ModelFactory
from deep_learning.callbacks import (
    BaseCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    CSVLoggerCallback,
    CallbackList,
)
from deep_learning.checkpoint_manager import CheckpointManager
from deep_learning.trainer import DLTrainer
from deep_learning.evaluator import DLEvaluationReport, DLEvaluator
from deep_learning.predictor import DLPredictor
from deep_learning.model_registry import DLModelRecord, DLModelRegistry
from deep_learning.experiment_tracker import DLExperimentRun, DLExperimentTracker
from deep_learning.export_manager import DLExportManager
from deep_learning.visualization import DLVisualizer
from deep_learning.report_generator import DLReportGenerator
from deep_learning.dl_engine import DLEngineConfig, DLEngineResult, DeepLearningEngine

__all__ = [
    # Sequence & Datasets
    "SequenceBuilder",
    "TimeSeriesDataset",
    "DatasetBuilder",
    "DLDataLoader",
    "DLPreprocessor",
    "TimeSeriesAugmenter",
    # Architectures & Model Factory
    "PyTorchBaseModel",
    "ModelFactory",
    # Callbacks & Checkpoint
    "BaseCallback",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "CSVLoggerCallback",
    "CallbackList",
    "CheckpointManager",
    # Training & Evaluation & Inference
    "DLTrainer",
    "DLEvaluationReport",
    "DLEvaluator",
    "DLPredictor",
    # MLOps Registry, Tracker & Export
    "DLModelRecord",
    "DLModelRegistry",
    "DLExperimentRun",
    "DLExperimentTracker",
    "DLExportManager",
    # Visual Analytics & Reports
    "DLVisualizer",
    "DLReportGenerator",
    # Engine
    "DLEngineConfig",
    "DLEngineResult",
    "DeepLearningEngine",
]
