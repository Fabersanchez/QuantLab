"""
QuantLab Feature Engineering Unit Tests.

Verifies functionality of FeatureMetadata, FeatureRegistry, FeatureGenerator,
FeatureValidator, FeatureScaler, FeatureEncoder, FeatureSelector, FeatureImportance,
FeaturePipeline, and FeatureEngine using standard library unittest.
"""

import unittest
import numpy as np
import pandas as pd

from machine_learning.features import (
    FeatureMetadata,
    FeatureRegistry,
    FeatureAlreadyRegisteredError,
    FeatureNotFoundError,
    FeatureGenerator,
    FeatureValidator,
    StandardScalerAdapter,
    MinMaxScalerAdapter,
    LabelEncoderAdapter,
    OneHotEncoderAdapter,
    VarianceThresholdSelector,
    CorrelationThresholdSelector,
    FeatureImportanceEvaluator,
    FeaturePipeline,
    FeatureEngine,
)


class TestQuantLabFeatureEngine(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        timestamps = pd.date_range("2026-01-01 09:30", periods=50, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(50))
        high_prices = close_prices + np.abs(np.random.randn(50))
        low_prices = close_prices - np.abs(np.random.randn(50))
        open_prices = low_prices + (high_prices - low_prices) * np.random.rand(50)
        volume = 1000.0 + np.random.rand(50) * 500.0

        self.sample_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
                "category_col": ["A", "B"] * 25,
                "target": np.random.choice([0, 1], size=50),
            }
        )

    def test_registry(self) -> None:
        registry = FeatureRegistry()
        meta = FeatureMetadata(name="log_return", category="Price")
        registry.register(meta)

        self.assertTrue(registry.has("log_return"))
        self.assertEqual(registry.get("log_return").category, "Price")

        with self.assertRaises(FeatureAlreadyRegisteredError):
            registry.register(meta)

        removed = registry.unregister("log_return")
        self.assertEqual(removed.name, "log_return")
        self.assertFalse(registry.has("log_return"))

    def test_feature_generator(self) -> None:
        generator = FeatureGenerator()
        gen_df = generator.generate_all(self.sample_df)

        self.assertIn("log_return", gen_df.columns)
        self.assertIn("volume_pct_change", gen_df.columns)
        self.assertIn("day_of_week", gen_df.columns)

    def test_feature_validator(self) -> None:
        validator = FeatureValidator(max_correlation_threshold=0.99)
        report = validator.validate(
            self.sample_df, ignore_cols=["timestamp", "category_col"]
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_features, 6)

    def test_scalers(self) -> None:
        std_scaler = StandardScalerAdapter()
        scaled = std_scaler.fit_transform(self.sample_df[["open", "close"]])
        self.assertAlmostEqual(scaled["open"].mean(), 0.0, places=5)

        minmax = MinMaxScalerAdapter()
        scaled_minmax = minmax.fit_transform(self.sample_df[["open", "close"]])
        self.assertAlmostEqual(scaled_minmax["open"].min(), 0.0, places=5)
        self.assertAlmostEqual(scaled_minmax["open"].max(), 1.0, places=5)

    def test_encoders(self) -> None:
        label_enc = LabelEncoderAdapter()
        encoded = label_enc.fit_transform(self.sample_df[["category_col"]])
        self.assertIn(0, encoded["category_col"].values)

        onehot = OneHotEncoderAdapter()
        oh_encoded = onehot.fit_transform(self.sample_df[["category_col"]])
        self.assertIn("category_col_A", oh_encoded.columns)

    def test_selectors(self) -> None:
        var_sel = VarianceThresholdSelector(threshold=0.01)
        selected = var_sel.select_features(self.sample_df[["open", "close"]])
        self.assertGreater(len(selected), 0)

        corr_sel = CorrelationThresholdSelector(max_correlation=0.99)
        selected_corr = corr_sel.select_features(self.sample_df[["open", "close"]])
        self.assertGreater(len(selected_corr), 0)

    def test_feature_importance(self) -> None:
        evaluator = FeatureImportanceEvaluator()
        report = evaluator.evaluate(
            self.sample_df[["open", "volume"]], self.sample_df["target"]
        )
        self.assertEqual(len(report.rankings), 2)

    def test_feature_engine_and_pipeline(self) -> None:
        engine = FeatureEngine()
        output_df, report = engine.run_pipeline(
            self.sample_df, target_col="target", timestamp_col="timestamp"
        )

        self.assertIn("timestamp", output_df.columns)
        self.assertIn("target", output_df.columns)
        self.assertGreater(output_df.shape[1], 3)
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()
