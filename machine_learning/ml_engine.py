"""
QuantLab Master Machine Learning Research Engine.

Orchestrates dataset management, feature store, target building, preprocessing pipelines,
feature selection & importance, cross-validation (including Purged TimeSeries CV), model training,
probability calibration, inference, model registry, experiment tracking, and reporting.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from core.logger import get_logger
from machine_learning.calibration import ProbabilityCalibrator
from machine_learning.cross_validation import CrossValidationFactory
from machine_learning.dataset_manager import DatasetManager, DatasetSplit
from machine_learning.evaluator import EvaluationReport, ModelEvaluator
from machine_learning.experiment_tracker import ExperimentTracker
from machine_learning.feature_importance import FeatureImportanceAnalyzer
from machine_learning.feature_selection import FeatureSelector
from machine_learning.feature_store import FeatureStore
from machine_learning.model_registry import ModelRecord, ModelRegistry
from machine_learning.predictor import Predictor
from machine_learning.preprocessing import PreprocessingPipeline
from machine_learning.target_builder import TargetBuilder
from machine_learning.trainer import ModelTrainer


logger = get_logger("MLEngine")


@dataclass
class MLEngineConfig:
    """Configuration specification for MLEngine execution."""

    model_type: str = "random_forest"
    target_type: str = "binary"  # 'binary', 'directional', 'return', 'triple_barrier'
    target_horizon: int = 1
    preprocessing_scaling: str = "standard"  # 'standard', 'minmax', 'robust', 'none'
    feature_selection_method: Optional[str] = "select_k_best"  # 'variance', 'select_k_best', 'rfe', 'lasso', 'boruta'
    n_selected_features: int = 10
    cv_type: str = "purged"  # 'purged', 'time_series', 'stratified', 'kfold'
    n_splits: int = 5
    calibrate_probabilities: bool = True
    calibration_method: str = "sigmoid"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    export_reports: bool = False
    author: str = "QuantLabML"


@dataclass
class MLEngineResult:
    """Dataclass encapsulating complete ML Research Lab pipeline results."""

    model_name: str
    model_id: str
    dataset_name: str
    model_obj: Any
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    selected_features: List[str]
    feature_importance: pd.Series
    evaluation_report: EvaluationReport
    model_record: ModelRecord
    execution_time_seconds: float = 0.0


class MLEngine:
    """Master Institutional Machine Learning Research Engine."""

    def __init__(self, config: Optional[MLEngineConfig] = None) -> None:
        """Initialize MLEngine."""
        self.config = config or MLEngineConfig()

        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()

        self._data: Optional[pd.DataFrame] = None
        self._asset_symbol: str = "GENERIC"
        self._feature_cols: List[str] = []
        self._target_col: Optional[str] = None
        self._fitted_pipeline: Optional[PreprocessingPipeline] = None
        self._active_model: Optional[Any] = None

    def load_data(self, df: pd.DataFrame, asset_symbol: str = "GENERIC") -> None:
        """Load market feature DataFrame."""
        self._data = df.copy()
        self._asset_symbol = asset_symbol
        logger.info(f"Loaded DataFrame into MLEngine: Asset='{asset_symbol}', Shape={df.shape}")

    def set_features_and_target(
        self, feature_cols: List[str], target_col: Optional[str] = None
    ) -> None:
        """Designate feature column names and optional target column name."""
        self._feature_cols = feature_cols
        self._target_col = target_col

    def start_pipeline(self) -> MLEngineResult:
        """Execute full end-to-end Machine Learning pipeline:
        Dataset Preparation -> Target Building -> Preprocessing -> Feature Selection -> CV Training -> Calibration -> Evaluation -> Registry & Experiment Logging.

        Returns:
            MLEngineResult dataclass.
        """
        if self._data is None:
            raise RuntimeError("No DataFrame loaded. Call load_data() first.")

        start_time = time.time()
        logger.info(f"Starting ML Research Pipeline: Model='{self.config.model_type}'...")

        df = self._data.copy()

        # 1. Build Target Variable if not explicitly provided
        if self._target_col is None or self._target_col not in df.columns:
            if self.config.target_type == "binary":
                y_target = TargetBuilder.build_binary_classification_target(df, horizon=self.config.target_horizon)
            elif self.config.target_type == "directional":
                y_target = TargetBuilder.build_directional_target(df, horizon=self.config.target_horizon)
            elif self.config.target_type == "triple_barrier":
                y_target = TargetBuilder.build_triple_barrier_target(df)
            else:
                y_target = TargetBuilder.build_continuous_return_target(df, horizon=self.config.target_horizon)

            target_name = y_target.name
            df = df.iloc[: len(y_target)]
            df[target_name] = y_target.values
            self._target_col = target_name

        # Extract features X and target y
        if not self._feature_cols:
            self._feature_cols = [c for c in df.columns if c != self._target_col]

        X_raw = df[self._feature_cols].copy()
        y = df[self._target_col].copy()

        # Register features into FeatureStore
        for col in self._feature_cols:
            self.feature_store.register_feature(col, X_raw[col], origin="MLEngineInput")

        # 2. Partition Dataset (Train / Val / Test)
        split: DatasetSplit = DatasetManager.train_test_split(
            X=X_raw, y=y, test_pct=0.2, val_pct=0.1, shuffle=False
        )

        # 3. Fit Preprocessing Pipeline
        self._fitted_pipeline = PreprocessingPipeline(scaling_method=self.config.preprocessing_scaling)
        X_train_proc = self._fitted_pipeline.fit_transform(split.X_train)
        X_val_proc = self._fitted_pipeline.transform(split.X_val) if split.X_val is not None else None
        X_test_proc = self._fitted_pipeline.transform(split.X_test) if split.X_test is not None else None

        # 4. Perform Feature Selection
        selected_features = list(X_train_proc.columns)
        if self.config.feature_selection_method == "variance":
            selected_features = FeatureSelector.select_variance_threshold(X_train_proc)
        elif self.config.feature_selection_method == "select_k_best":
            selected_features = FeatureSelector.select_k_best(
                X_train_proc, split.y_train, k=self.config.n_selected_features
            )
        elif self.config.feature_selection_method == "rfe":
            selected_features = FeatureSelector.select_rfe(
                X_train_proc, split.y_train, n_features_to_select=self.config.n_selected_features
            )
        elif self.config.feature_selection_method == "lasso":
            selected_features = FeatureSelector.select_lasso(X_train_proc, split.y_train)

        if not selected_features:
            selected_features = list(X_train_proc.columns)

        self._selected_features = selected_features
        X_train_sel = X_train_proc[selected_features]
        X_val_sel = X_val_proc[selected_features] if X_val_proc is not None else None
        X_test_sel = X_test_proc[selected_features] if X_test_proc is not None else None

        logger.info(f"Selected {len(selected_features)} features: {selected_features[:5]}...")

        # 5. Train Model via ModelTrainer
        trainer = ModelTrainer(model_type=self.config.model_type, params=self.config.hyperparameters)
        trained_model = trainer.train(X_train_sel, split.y_train)

        # 6. Calibrate Probabilities if enabled
        if self.config.calibrate_probabilities and X_val_sel is not None and split.y_val is not None:
            calibrated_model = ProbabilityCalibrator.calibrate(
                model=trained_model,
                X_val=X_val_sel,
                y_val=split.y_val,
                method=self.config.calibration_method,
            )
            self._active_model = calibrated_model
        else:
            self._active_model = trained_model

        # 7. Evaluate Model on Out-of-Sample Test Set
        eval_target_X = X_test_sel if X_test_sel is not None else X_train_sel
        eval_target_y = split.y_test if split.y_test is not None else split.y_train

        eval_report: EvaluationReport = ModelEvaluator.evaluate(
            model=self._active_model, X_test=eval_target_X, y_test=eval_target_y
        )

        # 8. Compute Feature Importance
        try:
            feat_imp = FeatureImportanceAnalyzer.calculate_tree_mdi(trained_model, selected_features)
        except Exception:
            feat_imp = FeatureImportanceAnalyzer.calculate_permutation_importance(
                self._active_model, eval_target_X, eval_target_y
            )

        # 9. Register Model into MLOps ModelRegistry
        model_record: ModelRecord = self.model_registry.register_model(
            name=self.config.model_type,
            model=self._active_model,
            hyperparameters=self.config.hyperparameters,
            metrics=eval_report.metrics,
            dataset_name=self._asset_symbol,
            author=self.config.author,
            status="EXPERIMENTAL",
        )

        # 10. Log Run into ExperimentTracker
        exec_duration = time.time() - start_time
        self.experiment_tracker.log_run(
            experiment_name=f"WFA_ML_{self.config.model_type}",
            model_type=self.config.model_type,
            hyperparameters=self.config.hyperparameters,
            metrics=eval_report.metrics,
            dataset_info={"asset": self._asset_symbol, "n_features": len(selected_features)},
            duration_seconds=exec_duration,
            author=self.config.author,
        )

        logger.info(
            f"ML Pipeline completed cleanly in {exec_duration:.2f}s! "
            f"ROC_AUC={eval_report.metrics.get('roc_auc', 0.5):.3f}"
        )

        return MLEngineResult(
            model_name=self.config.model_type,
            model_id=model_record.model_id,
            dataset_name=self._asset_symbol,
            model_obj=self._active_model,
            metrics=eval_report.metrics,
            hyperparameters=self.config.hyperparameters,
            selected_features=selected_features,
            feature_importance=feat_imp,
            evaluation_report=eval_report,
            model_record=model_record,
            execution_time_seconds=exec_duration,
        )

    def predict_signals(self, df: pd.DataFrame) -> pd.Series:
        """Run inference on new DataFrame and return quantitative trading signals (+1, 0, -1)."""
        if self._active_model is None or self._fitted_pipeline is None:
            raise RuntimeError("Model pipeline not trained. Call start_pipeline() first.")

        df_proc = self._fitted_pipeline.transform(df)
        if hasattr(self, "_selected_features") and self._selected_features:
            df_proc = df_proc[self._selected_features]
        return Predictor.predict_signal(self._active_model, df_proc)
