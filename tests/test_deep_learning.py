"""
QuantLab Deep Learning Research Lab Unit Tests.

Verifies complete functionality of SequenceBuilder, DatasetBuilder, DLDataLoader,
DLPreprocessor, TimeSeriesAugmenter, ModelFactory, Callbacks, CheckpointManager,
DLTrainer, DLEvaluator, DLPredictor, DLModelRegistry, DLExperimentTracker,
DLExportManager, DLVisualizer, DLReportGenerator, and master DeepLearningEngine.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from deep_learning import (
    SequenceBuilder,
    TimeSeriesDataset,
    DatasetBuilder,
    DLDataLoader,
    DLPreprocessor,
    TimeSeriesAugmenter,
    PyTorchBaseModel,
    ModelFactory,
    BaseCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    CallbackList,
    CheckpointManager,
    DLTrainer,
    DLEvaluationReport,
    DLEvaluator,
    DLPredictor,
    DLModelRecord,
    DLModelRegistry,
    DLExperimentRun,
    DLExperimentTracker,
    DLExportManager,
    DLVisualizer,
    DLReportGenerator,
    DLEngineConfig,
    DLEngineResult,
    DeepLearningEngine,
)


class TestDeepLearningResearchLab(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 100
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.2) + 0.1
        low = close - np.abs(np.random.randn(n) * 0.2) - 0.1
        open_p = low + (high - low) * 0.5
        volume = np.random.randint(1000, 5000, size=n).astype(float)

        feat_1 = np.random.randn(n)
        feat_2 = np.random.randn(n) * 2.0

        self.df = pd.DataFrame(
            {
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "feature_1": feat_1,
                "feature_2": feat_2,
            }
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_sequence_and_dataset_builder(self) -> None:
        X_seq, y_target = SequenceBuilder.create_sliding_windows(
            self.df, sequence_length=20, feature_cols=["feature_1", "feature_2"]
        )
        self.assertEqual(X_seq.shape, (81, 20, 2))

        # Dataset Builder
        dataset = DatasetBuilder.build_dataset_from_dataframe(
            self.df, sequence_length=20, feature_cols=["feature_1", "feature_2"]
        )
        self.assertEqual(dataset.n_samples, 81)
        self.assertEqual(dataset.sequence_length, 20)
        self.assertEqual(dataset.n_features, 2)

    def test_dataloader_and_preprocessing(self) -> None:
        dataset = DatasetBuilder.build_dataset_from_dataframe(
            self.df, sequence_length=20, feature_cols=["feature_1", "feature_2"]
        )

        # DLPreprocessor
        prep = DLPreprocessor(scaling_method="standard")
        proc_X = prep.fit_transform(dataset.X_seq)
        self.assertEqual(proc_X.shape, dataset.X_seq.shape)

        # DataLoader
        loader = DLDataLoader(dataset, batch_size=16, shuffle=False)
        self.assertGreater(len(loader), 0)
        for X_b, _ in loader:
            self.assertEqual(X_b.shape[1], 20)
            break

    def test_augmentation(self) -> None:
        X_seq = np.random.randn(10, 20, 3)
        y = np.random.randint(0, 2, size=10)

        noisy = TimeSeriesAugmenter.inject_noise(X_seq, noise_std=0.05)
        self.assertEqual(noisy.shape, X_seq.shape)

        scaled = TimeSeriesAugmenter.scale_magnitude(X_seq)
        self.assertEqual(scaled.shape, X_seq.shape)

        warped = TimeSeriesAugmenter.time_warp(X_seq)
        self.assertEqual(warped.shape, X_seq.shape)

        perm = TimeSeriesAugmenter.permute_blocks(X_seq, n_blocks=2)
        self.assertEqual(perm.shape, X_seq.shape)

        X_mix, y_mix = TimeSeriesAugmenter.apply_mixup(X_seq, y)
        self.assertEqual(X_mix.shape, X_seq.shape)

    def test_model_factory(self) -> None:
        architectures = ["mlp", "cnn", "lstm", "bilstm", "gru", "transformer", "hybrid"]
        for arch in architectures:
            model = ModelFactory.create_model(arch, input_dim=3, hidden_dim=16, output_dim=1)
            self.assertIsNotNone(model)
            dummy_input = np.random.randn(4, 20, 3).astype(np.float32)
            out = model(dummy_input)
            self.assertIsNotNone(out)

    def test_callbacks_and_checkpoint_manager(self) -> None:
        cb_list = CallbackList([EarlyStoppingCallback(patience=2)])
        stop = cb_list.on_epoch_end(1, {"val_loss": 0.5})
        self.assertFalse(stop)

        # Checkpoint Manager
        model = ModelFactory.create_model("lstm", input_dim=3)
        ckpt_path = os.path.join(self.temp_dir, "model.ckpt")
        saved = CheckpointManager.save_checkpoint(model, ckpt_path, epoch=1)
        self.assertTrue(os.path.exists(saved))

        loaded = CheckpointManager.load_checkpoint(saved)
        self.assertIsNotNone(loaded)

    def test_trainer_evaluator_and_predictor(self) -> None:
        dataset = DatasetBuilder.build_dataset_from_dataframe(
            self.df, sequence_length=15, target_col="close", feature_cols=["feature_1", "feature_2"]
        )
        loader = DLDataLoader(dataset, batch_size=16)

        model = ModelFactory.create_model("lstm", input_dim=2, hidden_dim=16)
        trainer = DLTrainer(model=model, lr=0.01)
        loss_hist = trainer.train_epochs(loader, epochs=2)

        self.assertIn("train_loss", loss_hist)
        self.assertEqual(len(loss_hist["train_loss"]), 2)

        # Evaluator
        eval_report = DLEvaluator.evaluate(model, loader)
        self.assertIn("accuracy", eval_report.metrics)

        # Predictor
        probas = DLPredictor.predict_proba(model, dataset.X_seq)
        self.assertEqual(len(probas), dataset.n_samples)

        signals = DLPredictor.predict_signal(model, dataset.X_seq)
        self.assertEqual(len(signals), dataset.n_samples)

    def test_registry_tracker_and_export_manager(self) -> None:
        # Registry
        reg = DLModelRegistry()
        model = ModelFactory.create_model("lstm", input_dim=2)
        rec = reg.register_model("lstm_model", model, metrics={"roc_auc": 0.82})
        self.assertEqual(rec.version, 1)

        # Tracker
        tracker = DLExperimentTracker()
        run = tracker.log_run("dl_exp", "lstm", {"hidden_dim": 16}, {"train_loss": [0.5]}, {"roc_auc": 0.82})
        self.assertEqual(run.model_type, "lstm")

        # Export Manager
        onnx_p = os.path.join(self.temp_dir, "model.onnx")
        DLExportManager.export_onnx(model, onnx_p)
        self.assertTrue(os.path.exists(onnx_p))

        ts_p = os.path.join(self.temp_dir, "model.pt")
        DLExportManager.export_torchscript(model, ts_p)
        self.assertTrue(os.path.exists(ts_p))

    def test_visualizer_and_report_generator(self) -> None:
        engine = DeepLearningEngine(DLEngineConfig(model_type="lstm", sequence_length=15, epochs=2))
        engine.load_data(self.df, asset_symbol="BTCUSD")
        res = engine.start_pipeline()

        # Visualizer
        viz = DLVisualizer()
        loss_svg = viz.generate_loss_curves_svg(res.loss_history)
        self.assertIn("<svg", loss_svg)

        # Report Generator
        reporter = DLReportGenerator(res)
        out_dir = os.path.join(self.temp_dir, "dl_reports")
        paths = reporter.export_all(out_dir)

        self.assertTrue(os.path.exists(paths["html"]))
        self.assertTrue(os.path.exists(paths["markdown"]))
        self.assertTrue(os.path.exists(paths["json"]))
        self.assertTrue(os.path.exists(paths["pdf"]))
        self.assertTrue(os.path.exists(paths["loss_history_csv"]))

    def test_dl_engine(self) -> None:
        config = DLEngineConfig(
            model_type="lstm",
            sequence_length=15,
            hidden_dim=16,
            batch_size=16,
            epochs=2,
        )
        engine = DeepLearningEngine(config)
        engine.load_data(self.df, asset_symbol="ETHUSD")

        res = engine.start_pipeline()
        self.assertIsInstance(res, DLEngineResult)
        self.assertGreater(res.execution_time_seconds, 0.0)

        # Inference Signals
        signals = engine.predict_signals(self.df)
        self.assertGreater(len(signals), 0)


if __name__ == "__main__":
    unittest.main()
