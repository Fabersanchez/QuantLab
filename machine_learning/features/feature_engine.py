"""
QuantLab Master Feature Engine.

Main orchestrator managing feature generation, validation, transformation,
registration, evaluation, and pipeline execution.
"""

from typing import List, Optional, Tuple
import pandas as pd

from machine_learning.features.feature_metadata import FeatureMetadata
from machine_learning.features.feature_registry import FeatureRegistry
from machine_learning.features.feature_generator import FeatureGenerator
from machine_learning.features.feature_validator import FeatureValidator, FeatureValidationReport
from machine_learning.features.feature_importance import FeatureImportanceEvaluator, FeatureImportanceReport
from machine_learning.features.feature_pipeline import FeaturePipeline


class FeatureEngine:
    """Master Feature Engineering Engine for QuantLab."""

    def __init__(self) -> None:
        """Initialize FeatureEngine subsystems."""
        self._registry = FeatureRegistry()
        self._generator = FeatureGenerator()
        self._validator = FeatureValidator()
        self._importance_evaluator = FeatureImportanceEvaluator()
        self._pipeline = FeaturePipeline()

    @property
    def registry(self) -> FeatureRegistry:
        """Access institutional FeatureRegistry."""
        return self._registry

    @property
    def generator(self) -> FeatureGenerator:
        """Access FeatureGenerator."""
        return self._generator

    @property
    def validator(self) -> FeatureValidator:
        """Access FeatureValidator."""
        return self._validator

    @property
    def importance_evaluator(self) -> FeatureImportanceEvaluator:
        """Access FeatureImportanceEvaluator."""
        return self._importance_evaluator

    @property
    def pipeline(self) -> FeaturePipeline:
        """Access FeaturePipeline."""
        return self._pipeline

    def register_feature_metadata(
        self,
        name: str,
        category: str = "General",
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> FeatureMetadata:
        """Register feature metadata record into the central registry."""
        meta = FeatureMetadata(
            name=name,
            category=category,
            description=description,
            dependencies=dependencies or [],
        )
        self._registry.register(meta, overwrite=True)
        return meta

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features using registered category generators."""
        return self._generator.generate_all(df)

    def validate_features(
        self, df: pd.DataFrame, ignore_cols: Optional[List[str]] = None
    ) -> FeatureValidationReport:
        """Validate feature dataset and return diagnostic report."""
        return self._validator.validate(df, ignore_cols=ignore_cols)

    def evaluate_importance(
        self, X: pd.DataFrame, y: pd.Series, top_k: int = 10
    ) -> FeatureImportanceReport:
        """Evaluate feature importance scores and rankings."""
        return self._importance_evaluator.evaluate(X, y, top_k=top_k)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, FeatureValidationReport]:
        """Execute full feature engineering pipeline end-to-end."""
        output_df, report = self._pipeline.run(
            df, target_col=target_col, timestamp_col=timestamp_col
        )

        # Register metadata for generated output columns
        for col in output_df.columns:
            if col not in (timestamp_col, target_col):
                if not self._registry.has(col):
                    self.register_feature_metadata(name=col, category="PipelineOutput")

        return output_df, report
