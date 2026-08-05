"""
QuantLab Master Data Engineering Platform Package.

Provides institutional quantitative data engineering: multi-source acquisition, schema validation,
data quality telemetry, missing value imputation, outlier clipping, Z-score scaling, multi-timeframe
resampling (1s to 1M), time-series synchronization, ML labeling (Triple Barrier), feature pipelines,
version snapshot rollbacks, storage (SQLite, DuckDB, Parquet), and multi-format exporters.
"""

from data_engine.cache import DataCache
from data_engine.cleaning import DataCleaner
from data_engine.data_engine import DataEngine
from data_engine.data_manager import DataManager
from data_engine.datasource import (
    BaseDataSource,
    CSVDataSource,
    ParquetDataSource,
    SQLiteDataSource,
)
from data_engine.datasource_registry import DataSourceRegistry
from data_engine.exporter import DataExporter
from data_engine.feature_pipeline import FeaturePipeline
from data_engine.ingestion import DataIngestionEngine
from data_engine.integrity import DataIntegrityChecker
from data_engine.labeling import DataLabeler
from data_engine.loaders import DataLoader
from data_engine.logger import DataEngineLogger, get_data_engine_logger
from data_engine.merger import DataMerger
from data_engine.metadata import DatasetMetadata
from data_engine.normalization import DataNormalizer
from data_engine.preprocessing import DataPreprocessor
from data_engine.resampling import DataResampler
from data_engine.schema import SchemaValidator
from data_engine.splitter import DataSplitter
from data_engine.storage import DataStorage
from data_engine.streaming import DataStreamer
from data_engine.synchronization import DataSynchronizer
from data_engine.validation import DataValidationReport, DataValidator
from data_engine.versioning import DatasetSnapshot, DataVersionManager

__all__ = [
    "DataEngine",
    "DataManager",
    "BaseDataSource",
    "CSVDataSource",
    "ParquetDataSource",
    "SQLiteDataSource",
    "DataSourceRegistry",
    "DataIngestionEngine",
    "DataLoader",
    "DataStreamer",
    "DataCleaner",
    "DataNormalizer",
    "DataPreprocessor",
    "DataValidator",
    "DataValidationReport",
    "SchemaValidator",
    "DataIntegrityChecker",
    "DataResampler",
    "DataSynchronizer",
    "DataMerger",
    "DataSplitter",
    "DataLabeler",
    "FeaturePipeline",
    "DataVersionManager",
    "DatasetSnapshot",
    "DataCache",
    "DataStorage",
    "DataExporter",
    "DatasetMetadata",
    "DataEngineLogger",
    "get_data_engine_logger",
]
