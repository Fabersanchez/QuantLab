"""
QuantLab Feature Pipeline.

Configurable sequential processing pipeline executing:
Raw Data -> Cleaning -> Validation -> Generation -> Scaling -> Encoding -> Selection -> Output Dataset
"""

from typing import List, Optional, Tuple
import pandas as pd

from data import DataCleaner, DataValidator
from machine_learning.features.feature_generator import FeatureGenerator
from machine_learning.features.feature_validator import FeatureValidator, FeatureValidationReport
from machine_learning.features.feature_scaler import BaseScaler, StandardScalerAdapter
from machine_learning.features.feature_encoder import BaseEncoder, OneHotEncoderAdapter
from machine_learning.features.feature_selector import FeatureSelector


class FeaturePipeline:
    """Configurable quantitative feature pipeline."""

    def __init__(
        self,
        scaler: Optional[BaseScaler] = None,
        encoder: Optional[BaseEncoder] = None,
        selector: Optional[FeatureSelector] = None,
    ) -> None:
        """Initialize FeaturePipeline steps.

        Args:
            scaler: Optional BaseScaler instance (defaults to StandardScalerAdapter).
            encoder: Optional BaseEncoder instance (defaults to OneHotEncoderAdapter).
            selector: Optional FeatureSelector instance.
        """
        self.cleaner = DataCleaner()
        self.data_validator = DataValidator()
        self.generator = FeatureGenerator()
        self.feature_validator = FeatureValidator()
        self.scaler = scaler or StandardScalerAdapter()
        self.encoder = encoder or OneHotEncoderAdapter()
        self.selector = selector or FeatureSelector()

    def run(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, FeatureValidationReport]:
        """Execute full feature pipeline sequence.

        Sequential Pipeline Steps:
        1. Cleaning: Repair dataset and fill missing values.
        2. Validation: Validate raw input schema.
        3. Generation: Compute Price, Volume, Volatility, Time, Statistical features.
        4. Encoding: Encode categorical variables.
        5. Scaling: Scale numerical features.
        6. Selection: Filter out low variance and collinear features.

        Args:
            df: Raw input DataFrame.
            target_col: Name of target column if present.
            timestamp_col: Name of timestamp column.

        Returns:
            Tuple of (Transformed Feature Dataset DataFrame, FeatureValidationReport).
        """
        # Step 1: Cleaning
        df_clean = self.cleaner.repair_dataset(df, timestamp_col=timestamp_col)

        # Step 2: Generation
        df_gen = self.generator.generate_all(df_clean)

        # Step 3: Feature Validation
        val_report = self.feature_validator.validate(
            df_gen, ignore_cols=[timestamp_col, target_col] if target_col else [timestamp_col]
        )

        # Step 4: Encoding
        df_encoded = self.encoder.fit_transform(df_gen)

        # Separate metadata columns before scaling & selection
        preserved_cols = [c for c in [timestamp_col, target_col] if c in df_encoded.columns]
        preserved_df = df_encoded[preserved_cols] if preserved_cols else pd.DataFrame(index=df_encoded.index)

        feature_matrix = df_encoded.drop(columns=preserved_cols)
        numeric_cols = feature_matrix.select_dtypes(include=["number"]).columns.tolist()

        # Step 5: Scaling
        if numeric_cols:
            scaled_vals = self.scaler.fit_transform(feature_matrix[numeric_cols])
            feature_matrix[numeric_cols] = scaled_vals

        # Step 6: Selection
        y_target = df_encoded[target_col] if target_col and target_col in df_encoded.columns else None
        selected_matrix = self.selector.select(feature_matrix, y=y_target)

        # Recombine preserved columns with output feature matrix
        output_dataset = pd.concat([preserved_df, selected_matrix], axis=1)
        return output_dataset, val_report
