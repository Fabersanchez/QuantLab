"""
QuantLab Master Deep Learning Research Engine.

Orchestrates 3D sequence creation, financial dataset building, dataloading, pre-processing,
data augmentation, neural model factory instantiation, callbacks (Early Stopping, Model Checkpoint),
epoch training loops, model evaluation, inference signal generation, MLOps model registry, and reporting.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.logger import get_logger
from deep_learning.augmentation import TimeSeriesAugmenter
from deep_learning.callbacks import CSVLoggerCallback, EarlyStoppingCallback, ModelCheckpointCallback
from deep_learning.dataloader import DLDataLoader
from deep_learning.dataset_builder import DatasetBuilder, TimeSeriesDataset
from deep_learning.evaluator import DLEvaluationReport, DLEvaluator
from deep_learning.experiment_tracker import DLExperimentTracker
from deep_learning.export_manager import DLExportManager
from deep_learning.model_factory import ModelFactory
from deep_learning.model_registry import DLModelRecord, DLModelRegistry
from deep_learning.predictor import DLPredictor
from deep_learning.preprocessing import DLPreprocessor
from deep_learning.trainer import DLTrainer


logger = get_logger("DeepLearningEngine")


@dataclass
class DLEngineConfig:
    """Configuration specification for DeepLearningEngine execution."""

    model_type: str = "lstm"  # 'mlp', 'cnn', 'lstm', 'bilstm', 'gru', 'transformer', 'hybrid'
    sequence_length: int = 30
    hidden_dim: int = 64
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 0.001
    target_type: str = "binary"  # 'binary', 'directional', 'return', 'triple_barrier'
    preprocessing_scaling: str = "standard"  # 'standard', 'minmax', 'robust', 'none'
    use_augmentation: bool = False
    early_stopping_patience: int = 3
    export_reports: bool = False
    author: str = "QuantLabDL"


@dataclass
class DLEngineResult:
    """Dataclass encapsulating complete Deep Learning pipeline outputs."""

    model_name: str
    model_id: str
    dataset_name: str
    model_obj: Any
    metrics: Dict[str, float]
    loss_history: Dict[str, List[float]]
    hyperparameters: Dict[str, Any]
    evaluation_report: DLEvaluationReport
    model_record: DLModelRecord
    execution_time_seconds: float = 0.0


class DeepLearningEngine:
    """Master Institutional Deep Learning Engine."""

    def __init__(self, config: Optional[DLEngineConfig] = None) -> None:
        """Initialize DeepLearningEngine."""
        self.config = config or DLEngineConfig()

        self.model_registry = DLModelRegistry()
        self.experiment_tracker = DLExperimentTracker()

        self._data: Optional[pd.DataFrame] = None
        self._asset_symbol: str = "GENERIC"
        self._feature_cols: List[str] = []
        self._target_col: Optional[str] = None
        self._fitted_preprocessor: Optional[DLPreprocessor] = None
        self._active_model: Optional[Any] = None

    def load_data(self, df: pd.DataFrame, asset_symbol: str = "GENERIC") -> None:
        """Load market DataFrame."""
        self._data = df.copy()
        self._asset_symbol = asset_symbol
        logger.info(f"Loaded DataFrame into DeepLearningEngine: Asset='{asset_symbol}', Shape={df.shape}")

    def set_features_and_target(
        self, feature_cols: List[str], target_col: Optional[str] = None
    ) -> None:
        """Set feature columns and optional target column."""
        self._feature_cols = feature_cols
        self._target_col = target_col

    def start_pipeline(self) -> DLEngineResult:
        """Execute full end-to-end Deep Learning research pipeline:
        3D Sequence Building -> Preprocessing -> Augmentation -> Model Factory -> Callbacks & Training -> Evaluation -> Model Registry & Experiment Logging.

        Returns:
            DLEngineResult dataclass.
        """
        if self._data is None:
            raise RuntimeError("No DataFrame loaded into DeepLearningEngine. Call load_data() first.")

        start_time = time.time()
        logger.info(f"Starting Deep Learning Pipeline: Architecture='{self.config.model_type}'...")

        df = self._data.copy()

        # 1. Build Target Column if not present
        if self._target_col is None or self._target_col not in df.columns:
            from machine_learning.target_builder import TargetBuilder
            if self.config.target_type == "binary":
                y_t = TargetBuilder.build_binary_classification_target(df)
            elif self.config.target_type == "directional":
                y_t = TargetBuilder.build_directional_target(df)
            elif self.config.target_type == "triple_barrier":
                y_t = TargetBuilder.build_triple_barrier_target(df)
            else:
                y_t = TargetBuilder.build_continuous_return_target(df)

            target_name = y_t.name
            df = df.iloc[: len(y_t)]
            df[target_name] = y_t.values
            self._target_col = target_name

        if not self._feature_cols:
            self._feature_cols = [c for c in df.columns if c != self._target_col]

        # 2. Build 3D Sequence Dataset
        dataset: TimeSeriesDataset = DatasetBuilder.build_dataset_from_dataframe(
            df=df,
            sequence_length=self.config.sequence_length,
            target_col=self._target_col,
            feature_cols=self._feature_cols,
        )

        # 3. Fit Sequence Preprocessor
        self._fitted_preprocessor = DLPreprocessor(scaling_method=self.config.preprocessing_scaling)
        X_seq_proc = self._fitted_preprocessor.fit_transform(dataset.X_seq)
        dataset.X_seq = X_seq_proc

        # 4. Optional Time Series Data Augmentation
        if self.config.use_augmentation:
            dataset.X_seq = TimeSeriesAugmenter.inject_noise(dataset.X_seq, noise_std=0.02)

        # Train / Validation Split (80% Train, 20% Validation)
        n = dataset.n_samples
        n_train = int(n * 0.8)

        train_ds = TimeSeriesDataset(
            X_seq=dataset.X_seq[:n_train],
            y_target=dataset.y_target[:n_train] if dataset.y_target is not None else None,
            feature_names=dataset.feature_names,
        )
        val_ds = TimeSeriesDataset(
            X_seq=dataset.X_seq[n_train:],
            y_target=dataset.y_target[n_train:] if dataset.y_target is not None else None,
            feature_names=dataset.feature_names,
        )

        train_loader = DLDataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)
        val_loader = DLDataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        # 5. Instantiate Neural Architecture via ModelFactory
        n_feats = dataset.n_features
        self._active_model = ModelFactory.create_model(
            model_type=self.config.model_type,
            input_dim=n_feats,
            hidden_dim=self.config.hidden_dim,
            output_dim=1,
        )

        # Setup Callbacks
        early_stop = EarlyStoppingCallback(patience=self.config.early_stopping_patience)
        callbacks = [early_stop]

        # 6. Execute DLTrainer Epochs Loop
        trainer = DLTrainer(
            model=self._active_model,
            lr=self.config.learning_rate,
            callbacks=callbacks,
        )

        loss_hist = trainer.train_epochs(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config.epochs,
        )

        # 7. Evaluate Model on Out-of-Sample Validation Data
        eval_report: DLEvaluationReport = DLEvaluator.evaluate(self._active_model, val_loader)

        # 8. Register Model in DLModelRegistry
        model_record: DLModelRecord = self.model_registry.register_model(
            name=self.config.model_type,
            model=self._active_model,
            hyperparameters={
                "sequence_length": self.config.sequence_length,
                "hidden_dim": self.config.hidden_dim,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
            metrics=eval_report.metrics,
            framework="PyTorch",
            dataset_name=self._asset_symbol,
            author=self.config.author,
            status="EXPERIMENTAL",
        )

        # 9. Log Run into DLExperimentTracker
        exec_duration = time.time() - start_time
        self.experiment_tracker.log_run(
            experiment_name=f"DL_{self.config.model_type}",
            model_type=self.config.model_type,
            hyperparameters={"seq_len": self.config.sequence_length, "hidden_dim": self.config.hidden_dim},
            loss_history=loss_hist,
            metrics=eval_report.metrics,
            dataset_info={"asset": self._asset_symbol, "n_samples": n},
            duration_seconds=exec_duration,
            author=self.config.author,
        )

        logger.info(
            f"Deep Learning Pipeline completed cleanly in {exec_duration:.2f}s! "
            f"ROC_AUC={eval_report.metrics.get('roc_auc', 0.5):.3f}"
        )

        return DLEngineResult(
            model_name=self.config.model_type,
            model_id=model_record.model_id,
            dataset_name=self._asset_symbol,
            model_obj=self._active_model,
            metrics=eval_report.metrics,
            loss_history=loss_hist,
            hyperparameters={"seq_len": self.config.sequence_length, "hidden_dim": self.config.hidden_dim},
            evaluation_report=eval_report,
            model_record=model_record,
            execution_time_seconds=exec_duration,
        )

    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Run inference on new market DataFrame and return quantitative trading signals (+1, 0, -1)."""
        if self._active_model is None or self._fitted_preprocessor is None:
            raise RuntimeError("Deep Learning pipeline not trained. Call start_pipeline() first.")

        dataset = DatasetBuilder.build_dataset_from_dataframe(
            df=df,
            sequence_length=self.config.sequence_length,
            feature_cols=self._feature_cols,
        )

        X_seq_proc = self._fitted_preprocessor.transform(dataset.X_seq)
        return DLPredictor.predict_signal(self._active_model, X_seq_proc)
