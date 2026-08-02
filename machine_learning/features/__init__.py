"""QuantLab Machine Learning Feature Engineering Package."""

from machine_learning.features.feature_metadata import FeatureMetadata
from machine_learning.features.feature_registry import (
    FeatureRegistry,
    FeatureAlreadyRegisteredError,
    FeatureNotFoundError,
)
from machine_learning.features.feature_generator import (
    FeatureGenerator,
    BaseFeatureGenerator,
    PriceFeatureGenerator,
    VolumeFeatureGenerator,
    VolatilityFeatureGenerator,
    TimeFeatureGenerator,
    StatisticalFeatureGenerator,
)
from machine_learning.features.feature_validator import (
    FeatureValidator,
    FeatureValidationReport,
)
from machine_learning.features.feature_scaler import (
    BaseScaler,
    StandardScalerAdapter,
    MinMaxScalerAdapter,
    RobustScalerAdapter,
    NormalizerAdapter,
    ScalerFactory,
)
from machine_learning.features.feature_encoder import (
    BaseEncoder,
    LabelEncoderAdapter,
    OneHotEncoderAdapter,
    TargetEncoderAdapter,
)
from machine_learning.features.feature_selector import (
    BaseFeatureSelector,
    VarianceThresholdSelector,
    CorrelationThresholdSelector,
    MutualInformationSelector,
    FeatureSelector,
)
from machine_learning.features.feature_importance import (
    FeatureImportanceEvaluator,
    FeatureImportanceReport,
)
from machine_learning.features.feature_pipeline import FeaturePipeline
from machine_learning.features.feature_engine import FeatureEngine

__all__ = [
    "FeatureMetadata",
    "FeatureRegistry",
    "FeatureAlreadyRegisteredError",
    "FeatureNotFoundError",
    "FeatureGenerator",
    "BaseFeatureGenerator",
    "PriceFeatureGenerator",
    "VolumeFeatureGenerator",
    "VolatilityFeatureGenerator",
    "TimeFeatureGenerator",
    "StatisticalFeatureGenerator",
    "FeatureValidator",
    "FeatureValidationReport",
    "BaseScaler",
    "StandardScalerAdapter",
    "MinMaxScalerAdapter",
    "RobustScalerAdapter",
    "NormalizerAdapter",
    "ScalerFactory",
    "BaseEncoder",
    "LabelEncoderAdapter",
    "OneHotEncoderAdapter",
    "TargetEncoderAdapter",
    "BaseFeatureSelector",
    "VarianceThresholdSelector",
    "CorrelationThresholdSelector",
    "MutualInformationSelector",
    "FeatureSelector",
    "FeatureImportanceEvaluator",
    "FeatureImportanceReport",
    "FeaturePipeline",
    "FeatureEngine",
]
