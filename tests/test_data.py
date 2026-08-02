"""
QuantLab Data Engine Unit Tests.

Verifies functionality of BaseDataSource, DataLoader, DataValidator,
DataCleaner, DataTransformer, MarketDataset, MemoryCache, and Storage adapters.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from data import (
    MockDataSource,
    DataLoader,
    DataValidator,
    DataCleaner,
    DataTransformer,
    MarketDataset,
    MemoryCache,
    CSVStorage,
    SQLiteStorage,
    StorageFactory,
)


class TestQuantLabDataEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        timestamps = pd.date_range("2026-01-01 09:30", periods=5, freq="1min")
        self.sample_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0, 106.0, 107.0, 108.0, 109.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [102.0, 103.0, 104.0, 105.0, 106.0],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_datasource_and_loader_api(self) -> None:
        mock_ds = MockDataSource()
        loader = DataLoader(datasource=mock_ds)
        df = loader.load_api("EURUSD", "1m", limit=10)
        self.assertEqual(len(df), 10)
        self.assertIn("close", df.columns)

    def test_data_validator_pass(self) -> None:
        validator = DataValidator()
        report = validator.validate_ohlcv(self.sample_df)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.invalid_ohlc_count, 0)
        self.assertEqual(report.duplicate_timestamps, 0)

    def test_data_validator_fail(self) -> None:
        bad_df = self.sample_df.copy()
        bad_df.loc[0, "high"] = 90.0  # Invalid High < Low
        validator = DataValidator()
        report = validator.validate_ohlcv(bad_df)
        self.assertFalse(report.is_valid)
        self.assertGreater(report.invalid_ohlc_count, 0)

    def test_data_cleaner(self) -> None:
        dirty_df = pd.concat([self.sample_df, self.sample_df.iloc[[0]]])  # Dup
        cleaned = DataCleaner.repair_dataset(dirty_df)
        self.assertEqual(len(cleaned), len(self.sample_df))

    def test_data_transformer_splits_and_scaling(self) -> None:
        train, test = DataTransformer.train_test_split(self.sample_df, train_ratio=0.6)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)

        norm = DataTransformer.normalize(self.sample_df["close"])
        self.assertAlmostEqual(norm.min(), 0.0)
        self.assertAlmostEqual(norm.max(), 1.0)

        windows = DataTransformer.create_rolling_windows(
            self.sample_df["close"].values, window_size=3, step=1
        )
        self.assertEqual(len(windows), 3)

    def test_market_dataset(self) -> None:
        dataset = MarketDataset(
            data=self.sample_df,
            asset="BTC/USD",
            timeframe="1m",
            broker="Binance",
            features=["open", "high", "low", "close"],
            target="close",
        )
        self.assertEqual(dataset.metadata.asset, "BTC/USD")
        self.assertEqual(dataset.rows, 5)
        stats = dataset.summary_statistics()
        self.assertIn("means", stats)

    def test_memory_cache(self) -> None:
        cache = MemoryCache(max_size=2)
        cache.set("key1", "val1")
        cache.set("key2", "val2")
        self.assertEqual(cache.get("key1"), "val1")

        # Evict key2 when adding key3 after key1 was accessed
        cache.set("key3", "val3")
        self.assertIsNone(cache.get("key2"))
        self.assertEqual(cache.get("key3"), "val3")

    def test_storage(self) -> None:
        csv_path = os.path.join(self.temp_dir, "test.csv")
        csv_storage = CSVStorage()
        csv_storage.save(self.sample_df, csv_path)
        loaded_csv = csv_storage.load(csv_path)
        self.assertEqual(len(loaded_csv), 5)

        db_path = os.path.join(self.temp_dir, "test.db")
        sqlite_storage = StorageFactory.get_storage("sqlite")
        sqlite_storage.save(self.sample_df, db_path, table_name="ohlcv")
        loaded_sqlite = sqlite_storage.load(db_path, table_name="ohlcv")
        self.assertEqual(len(loaded_sqlite), 5)


if __name__ == "__main__":
    unittest.main()
