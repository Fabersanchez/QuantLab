"""
QuantLab Machine Learning Research Lab Unit Tests.

Verifies complete functionality of DatasetManager, FeatureStore, PreprocessingPipeline,
TargetBuilder, FeatureSelector, FeatureImportanceAnalyzer, ModelRegistry, ModelManager,
CrossValidation (including Purged TimeSeries CV), ModelTrainer, ModelEvaluator, Predictor,
ProbabilityCalibrator, EnsembleEngine, ModelExplainer, ExperimentTracker, MLVisualizer,
MLReportGenerator, and master MLEngine using standard library unittest.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from machine_learning import (
    DatasetSplit,
    DatasetManager,
    FeatureMetadata,
    FeatureStore,
    PreprocessingPipeline,
    TargetBuilder,
    FeatureSelector,
    FeatureImportanceAnalyzer,
    ModelRecord,
    ModelRegistry,
    ModelManager,
    PurgedGroupTimeSeriesSplit,
    CrossValidationFactory,
    ModelTrainer,
    MLMetricsCalculator,
    EvaluationReport,
    ModelEvaluator,
    Predictor,
    ProbabilityCalibrator,
    EnsembleEngine,
    ModelExplainer,
    ExperimentRun,
    ExperimentTracker,
    HyperparameterManager,
    MLVisualizer,
    MLReportGenerator,
    MLEngineConfig,
    MLEngineResult,
    MLEngine,
)


class TestMachineLearningResearchLab(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 120
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.2) + 0.1
        low = close - np.abs(np.random.randn(n) * 0.2) - 0.1
        open_p = low + (high - low) * 0.5
        volume = np.random.randint(1000, 5000, size=n).astype(float)

        feat_1 = np.random.randn(n)
        feat_2 = np.random.randn(n) * 2.0 + 1.0
        feat_3 = np.random.randn(n) * 0.5

        self.df = pd.DataFrame(
            {
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "feature_1": feat_1,
                "feature_2": feat_2,
                "feature_3": feat_3,
            }
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_dataset_manager_and_feature_store(self) -> None:
        # Feature Store
        fs = FeatureStore()
        meta = fs.register_feature("feature_1", self.df["feature_1"], origin="Technical")
        self.assertEqual(meta.name, "feature_1")
        self.assertEqual(meta.version, 1)

        data, fetched_meta = fs.get_feature("feature_1")
        self.assertEqual(len(data), len(self.df))

        # Dataset Manager Split
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]

        split = DatasetManager.train_test_split(X, y_bin, test_pct=0.2, val_pct=0.1)
        self.assertIsNotNone(split.X_train)
        self.assertIsNotNone(split.X_val)
        self.assertIsNotNone(split.X_test)

    def test_preprocessing_and_target_builder(self) -> None:
        # Preprocessing
        prep = PreprocessingPipeline(scaling_method="standard", outlier_method="zscore")
        transformed = prep.fit_transform(self.df[["feature_1", "feature_2"]])
        self.assertAlmostEqual(transformed["feature_1"].mean(), 0.0, places=1)

        # Target Builder
        y_binary = TargetBuilder.build_binary_classification_target(self.df)
        self.assertEqual(len(y_binary), len(self.df) - 1)

        y_dir = TargetBuilder.build_directional_target(self.df)
        self.assertEqual(len(y_dir), len(self.df) - 1)

        y_tb = TargetBuilder.build_triple_barrier_target(self.df)
        self.assertGreater(len(y_tb), 0)

    def test_feature_selection_and_importance(self) -> None:
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]

        # Feature Selection
        sel_k = FeatureSelector.select_k_best(X, y_bin, k=2)
        self.assertLessEqual(len(sel_k), 2)

        sel_rfe = FeatureSelector.select_rfe(X, y_bin, n_features_to_select=2)
        self.assertLessEqual(len(sel_rfe), 2)

        # Feature Importance
        trainer = ModelTrainer(model_type="random_forest")
        model = trainer.train(X, y_bin)

        imp_mdi = FeatureImportanceAnalyzer.calculate_tree_mdi(model, list(X.columns))
        self.assertEqual(len(imp_mdi), 3)

        imp_perm = FeatureImportanceAnalyzer.calculate_permutation_importance(model, X, y_bin)
        self.assertEqual(len(imp_perm), 3)

    def test_cross_validation_and_purged_split(self) -> None:
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]

        # Purged CV (De Prado)
        purged_cv = PurgedGroupTimeSeriesSplit(n_splits=3, purge_bars=2, embargo_bars=2)
        splits = list(purged_cv.split(X, y_bin))
        self.assertEqual(len(splits), 3)

        # Factory
        ts_cv = CrossValidationFactory.create("time_series", n_splits=3)
        self.assertIsNotNone(ts_cv)

    def test_model_training_and_prediction(self) -> None:
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]

        trainer = ModelTrainer(model_type="random_forest", params={"n_estimators": 20})
        model = trainer.train(X, y_bin)

        preds = Predictor.predict(model, X)
        self.assertEqual(len(preds), len(X))

        probas = Predictor.predict_proba(model, X)
        self.assertEqual(probas.shape[0], len(X))

        signals = Predictor.predict_signal(model, X)
        self.assertEqual(len(signals), len(X))

    def test_calibration_and_ensembles(self) -> None:
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]

        trainer = ModelTrainer(model_type="random_forest")
        model = trainer.train(X.iloc[:80], y_bin.iloc[:80])

        # Calibration
        calibrated = ProbabilityCalibrator.calibrate(model, X.iloc[80:], y_bin.iloc[80:], method="sigmoid")
        self.assertIsNotNone(calibrated)

        # Ensemble
        m1 = ModelTrainer("random_forest").train(X, y_bin)
        m2 = ModelTrainer("extra_trees").train(X, y_bin)
        voting = EnsembleEngine.create_voting_ensemble([("rf", m1), ("et", m2)], voting_type="soft")
        voting.fit(X, y_bin)
        self.assertIsNotNone(voting)

    def test_model_registry_and_manager(self) -> None:
        reg = ModelRegistry()
        m = ModelTrainer("random_forest").train(self.df[["feature_1"]].iloc[:50], pd.Series([1]*25 + [0]*25))

        record = reg.register_model("rf_model", m, hyperparameters={"n_estimators": 10}, metrics={"roc_auc": 0.85})
        self.assertEqual(record.version, 1)
        self.assertEqual(record.status, "EXPERIMENTAL")

        reg.update_status(record.model_id, "STAGING")
        self.assertEqual(reg.get_model(record.model_id).status, "STAGING")

        # Model Manager save & load
        save_path = os.path.join(self.temp_dir, "model.joblib")
        ModelManager.save_model(m, save_path)
        loaded = ModelManager.load_model(save_path)
        self.assertIsNotNone(loaded)

    def test_explainability_and_experiment_tracker(self) -> None:
        y_bin = TargetBuilder.build_binary_classification_target(self.df)
        X = self.df.iloc[: len(y_bin)][["feature_1", "feature_2", "feature_3"]]
        model = ModelTrainer("random_forest").train(X, y_bin)

        # Explainability
        shap_res = ModelExplainer.calculate_shap_values(model, X.iloc[:5])
        self.assertIn("values", shap_res)

        path = ModelExplainer.get_decision_path(model, X, row_index=0)
        self.assertGreater(len(path), 0)

        # Experiment Tracker
        tracker = ExperimentTracker()
        run = tracker.log_run("exp1", "random_forest", {"n_estimators": 50}, {"roc_auc": 0.88})
        self.assertEqual(run.experiment_name, "exp1")
        self.assertIsNotNone(tracker.get_best_run("roc_auc"))

    def test_visualizer_and_report_generator(self) -> None:
        engine = MLEngine(MLEngineConfig(model_type="random_forest", target_type="binary"))
        engine.load_data(self.df, asset_symbol="EURUSD")
        res = engine.start_pipeline()

        # Visualizer
        viz = MLVisualizer()
        cm_svg = viz.generate_confusion_matrix_svg(res.evaluation_report.confusion_matrix)
        self.assertIn("<svg", cm_svg)

        # Report Generator
        reporter = MLReportGenerator(res)
        out_dir = os.path.join(self.temp_dir, "ml_reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["feature_importance_csv"]))

    def test_ml_engine(self) -> None:
        config = MLEngineConfig(
            model_type="random_forest",
            target_type="binary",
            feature_selection_method="select_k_best",
            n_selected_features=3,
        )
        engine = MLEngine(config)
        engine.load_data(self.df, asset_symbol="EURUSD")

        res = engine.start_pipeline()
        self.assertIsInstance(res, MLEngineResult)
        self.assertGreater(res.execution_time_seconds, 0.0)

        # Test inference signals
        signals = engine.predict_signals(self.df)
        self.assertEqual(len(signals), len(self.df))


if __name__ == "__main__":
    unittest.main()
