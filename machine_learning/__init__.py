"""
QuantLab Institutional Machine Learning Research Lab Package.

Provides complete MLOps and quantitative Machine Learning infrastructure:
Dataset management, Feature Store, Preprocessing pipelines, Target Builders (Triple Barrier),
Feature Selection & Importance (MDI, MDA, Permutation, SFI, Boruta), Model Registry & Manager,
Cross-Validation (including De Prado Purged & Embargoed Time Series CV), Model Trainers
(scikit-learn, XGBoost, LightGBM, CatBoost, SVM, MLP), Probability Calibration, Prediction Inference Engine,
Ensemble Engine (Voting, Stacking, Bagging, Blending), Explainability (SHAP, LIME, Decision Path),
Automated MLOps Experiment Tracking, Visual Analytics, Multi-Format Reporting, and master MLEngine.
"""

from machine_learning.dataset_manager import (
    DatasetSplit,
    DatasetManager,
)
from machine_learning.feature_store import (
    FeatureMetadata,
    FeatureStore,
)
from machine_learning.preprocessing import (
    PreprocessingPipeline,
)
from machine_learning.target_builder import (
    TargetBuilder,
)
from machine_learning.feature_selection import (
    FeatureSelector,
)
from machine_learning.feature_importance import (
    FeatureImportanceAnalyzer,
)
from machine_learning.model_registry import (
    ModelRecord,
    ModelRegistry,
)
from machine_learning.model_manager import (
    ModelManager,
)
from machine_learning.cross_validation import (
    PurgedGroupTimeSeriesSplit,
    CrossValidationFactory,
)
from machine_learning.trainer import (
    ModelTrainer,
)
from machine_learning.metrics import (
    MLMetricsCalculator,
)
from machine_learning.evaluator import (
    EvaluationReport,
    ModelEvaluator,
)
from machine_learning.predictor import (
    Predictor,
)
from machine_learning.calibration import (
    ProbabilityCalibrator,
)
from machine_learning.ensemble_engine import (
    EnsembleEngine,
)
from machine_learning.explainability import (
    ModelExplainer,
)
from machine_learning.experiment_tracker import (
    ExperimentRun,
    ExperimentTracker,
)
from machine_learning.hyperparameter_manager import (
    HyperparameterManager,
)
from machine_learning.visualization import (
    MLVisualizer,
)
from machine_learning.report_generator import (
    MLReportGenerator,
)
from machine_learning.ml_engine import (
    MLEngineConfig,
    MLEngineResult,
    MLEngine,
)

__all__ = [
    # Data & Features
    "DatasetSplit",
    "DatasetManager",
    "FeatureMetadata",
    "FeatureStore",
    "PreprocessingPipeline",
    "TargetBuilder",
    "FeatureSelector",
    "FeatureImportanceAnalyzer",
    # Models & Registry
    "ModelRecord",
    "ModelRegistry",
    "ModelManager",
    "PurgedGroupTimeSeriesSplit",
    "CrossValidationFactory",
    "ModelTrainer",
    "MLMetricsCalculator",
    "EvaluationReport",
    "ModelEvaluator",
    "Predictor",
    "ProbabilityCalibrator",
    "EnsembleEngine",
    # Explainability & MLOps Tracking
    "ModelExplainer",
    "ExperimentRun",
    "ExperimentTracker",
    "HyperparameterManager",
    "MLVisualizer",
    "MLReportGenerator",
    # Engine
    "MLEngineConfig",
    "MLEngineResult",
    "MLEngine",
]
