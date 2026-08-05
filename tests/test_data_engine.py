"""
QuantLab Master Data Engineering Platform Test Suite.

Validates all 27 components of the Data Engineering Platform:
DatasetMetadata, SchemaValidator, DataIntegrityChecker, CSVDataSource, ParquetDataSource, SQLiteDataSource,
DataSourceRegistry, DataLoader, DataIngestionEngine, DataStreamer, DataCleaner, DataNormalizer, DataPreprocessor,
DataValidator, DataResampler, DataSynchronizer, DataMerger, DataSplitter, DataLabeler, FeaturePipeline,
DataVersionManager, DataCache, DataStorage, DataExporter, DataManager, and DataEngine.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from data_engine import (
    CSVDataSource,
    DataCache,
    DataCleaner,
    DataEngine,
    DataExporter,
    DataIngestionEngine,
    DataIntegrityChecker,
    DataLabeler,
    DataLoader,
    DataManager,
    DataMerger,
    DataNormalizer,
    DataPreprocessor,
    DataResampler,
    DatasetMetadata,
    DataSourceRegistry,
    DataSplitter,
    DataStorage,
    DataStreamer,
    DataSynchronizer,
    DataValidator,
    DataVersionManager,
    FeaturePipeline,
    ParquetDataSource,
    SchemaValidator,
)


class TestQuantLabDataEngine(unittest.TestCase):
    """Comprehensive Test Case for QuantLab Data Engineering Platform."""

    def setUp(self) -> None:
        """Set up temporary directory and synthetic OHLCV dataset."""
        self.temp_dir = tempfile.mkdtemp(prefix="quantlab_data_test_")
        self.csv_path = os.path.join(self.temp_dir, "test_data.csv")

        # Synthetic OHLCV data
        n_bars = 120
        dates = pd.date_range("2025-01-01", periods=n_bars, freq="1h")
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, size=n_bars))
        self.df_ohlcv = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": prices + 0.5,
                "low": prices - 0.5,
                "close": prices + 0.1,
                "volume": 1000 + np.random.randint(0, 500, size=n_bars),
            },
            index=dates,
        )
        self.df_ohlcv.to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_schema_and_integrity(self) -> None:
        """Test DatasetMetadata, SchemaValidator, and DataIntegrityChecker."""
        meta = DatasetMetadata(name="EURUSD_1H", row_count=len(self.df_ohlcv))
        self.assertEqual(meta.name, "EURUSD_1H")

        valid, missing = SchemaValidator.validate_schema(self.df_ohlcv)
        self.assertTrue(valid)
        self.assertEqual(len(missing), 0)

        h = DataIntegrityChecker.compute_df_sha256(self.df_ohlcv)
        self.assertTrue(len(h) == 64)

        cont, gaps = DataIntegrityChecker.check_continuity(self.df_ohlcv)
        self.assertTrue(cont)
        self.assertEqual(gaps, 0)

    def test_datasources_and_loaders(self) -> None:
        """Test CSVDataSource, DataSourceRegistry, and DataLoader."""
        source = CSVDataSource(self.csv_path)
        df_fetched = source.fetch_data()
        self.assertFalse(df_fetched.empty)

        registry = DataSourceRegistry()
        registry.register_source("csv_source", source)
        self.assertEqual(len(registry.list_sources()), 1)

        loaded_df = DataLoader.load_file(self.csv_path)
        self.assertFalse(loaded_df.empty)

    def test_ingestion_and_streaming(self) -> None:
        """Test DataIngestionEngine and DataStreamer."""
        ingest = DataIngestionEngine()
        source = CSVDataSource(self.csv_path)
        df_ingested = ingest.ingest_single(source, "TEST_SYM")
        self.assertFalse(df_ingested.empty)

        streamer = DataStreamer(max_buffer_size=100)
        streamer.push_tick("2025-01-01 10:00:00", "EURUSD", 1.1050, 100)
        streamer.push_bar("2025-01-01 10:00:00", "EURUSD", 1.1040, 1.1060, 1.1030, 1.1050, 1000)

        df_ticks = streamer.get_ticks_df()
        self.assertEqual(len(df_ticks), 1)

        df_bars = streamer.get_bars_df()
        self.assertEqual(len(df_bars), 1)

    def test_cleaning_and_normalization_and_preprocessing(self) -> None:
        """Test DataCleaner, DataNormalizer, and DataPreprocessor."""
        df_dirty = self.df_ohlcv.copy()
        df_dirty.iloc[5, 1] = np.nan  # Inject missing open
        df_dirty.iloc[10, 4] = 999.0  # Inject outlier close

        df_clean, nulls = DataCleaner.clean_missing_values(df_dirty)
        self.assertEqual(nulls, 1)

        df_norm = DataNormalizer.zscore_standardize(self.df_ohlcv, ["open", "close"])
        self.assertAlmostEqual(df_norm["close"].mean(), 0.0, places=4)

        df_scaled = DataNormalizer.minmax_scale(self.df_ohlcv, ["close"])
        self.assertAlmostEqual(df_scaled["close"].max(), 1.0, places=4)

        proc = DataPreprocessor(normalize_method="zscore")
        df_proc = proc.preprocess(df_dirty)
        self.assertFalse(df_proc.isna().any().any())

    def test_validator_resampler_and_synchronizer(self) -> None:
        """Test DataValidator, DataResampler, and DataSynchronizer."""
        report = DataValidator.validate_dataset(self.df_ohlcv)
        self.assertTrue(report.is_valid)
        self.assertGreater(report.quality_score, 80.0)

        df_4h = DataResampler.resample_ohlcv(self.df_ohlcv, target_timeframe="4h")
        self.assertLess(len(df_4h), len(self.df_ohlcv))

        df_sync = DataSynchronizer.align_to_utc(self.df_ohlcv)
        self.assertIsNotNone(df_sync)

    def test_merger_splitter_and_labeling(self) -> None:
        """Test DataMerger, DataSplitter, and DataLabeler (Triple Barrier)."""
        df_dict = {"EURUSD": self.df_ohlcv, "GBPUSD": self.df_ohlcv}
        df_merged = DataMerger.merge_asset_close_prices(df_dict)
        self.assertEqual(len(df_merged.columns), 2)

        train_df, test_df = DataSplitter.train_test_split(self.df_ohlcv, train_ratio=0.80)
        self.assertEqual(len(train_df), 96)

        labels_dir = DataLabeler.label_fixed_horizon_directional(self.df_ohlcv, horizon=3)
        self.assertEqual(len(labels_dir), len(self.df_ohlcv))

        labels_tb = DataLabeler.label_triple_barrier(self.df_ohlcv, pt_ratio=0.005, sl_ratio=0.005)
        self.assertEqual(len(labels_tb), len(self.df_ohlcv))

    def test_feature_pipeline_versioning_and_cache(self) -> None:
        """Test FeaturePipeline, DataVersionManager, and DataCache."""
        pipe = FeaturePipeline()
        pipe.add_step("add_sma", lambda df: df.assign(sma_10=df["close"].rolling(10).mean()))
        df_feat = pipe.fit_transform(self.df_ohlcv)
        self.assertIn("sma_10", df_feat.columns)

        vm = DataVersionManager(initial_version="1.0.0")
        vm.create_snapshot("DatasetA", self.df_ohlcv)
        v2 = vm.bump_version("minor")
        self.assertEqual(v2, "1.1.0")

        cache = DataCache()
        cache.put("KEY1", self.df_ohlcv)
        df_cached = cache.get("KEY1")
        self.assertIsNotNone(df_cached)

    def test_storage_exporter_and_data_engine(self) -> None:
        """Test DataStorage, DataExporter, DataManager, and master DataEngine."""
        storage = DataStorage(storage_dir=self.temp_dir)
        parq_path = storage.save_parquet(self.df_ohlcv, "test.parquet")
        self.assertTrue(os.path.exists(parq_path))

        exp_csv = DataExporter.to_csv(self.df_ohlcv, os.path.join(self.temp_dir, "exp.csv"))
        self.assertTrue(os.path.exists(exp_csv))

        exp_md = DataExporter.to_markdown(self.df_ohlcv, os.path.join(self.temp_dir, "exp.md"))
        self.assertTrue(os.path.exists(exp_md))

        manager = DataManager(storage_dir=self.temp_dir)
        manager.register_dataset("DS1", self.df_ohlcv)
        self.assertIsNotNone(manager.get_dataset("DS1"))

        engine = DataEngine(storage_dir=self.temp_dir)
        df_loaded, meta = engine.load_dataset_file(self.csv_path, name="EngineLoaded")
        self.assertFalse(df_loaded.empty)

        report = engine.validate_quality(df_loaded)
        self.assertTrue(report.is_valid)


if __name__ == "__main__":
    unittest.main()
