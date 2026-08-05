"""
QuantLab Master Data Engineering Platform Engine.

Centralizes data acquisition, schema validation, data quality telemetry, cleaning, standardization,
multi-timeframe resampling, ML labeling (Triple Barrier), feature pipeline transformation,
versioning, storage, and multi-format exporting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from data_engine.cache import DataCache
from data_engine.cleaning import DataCleaner
from data_engine.data_manager import DataManager
from data_engine.datasource import BaseDataSource, CSVDataSource, ParquetDataSource
from data_engine.datasource_registry import DataSourceRegistry
from data_engine.exporter import DataExporter
from data_engine.feature_pipeline import FeaturePipeline
from data_engine.ingestion import DataIngestionEngine
from data_engine.labeling import DataLabeler
from data_engine.logger import get_data_engine_logger
from data_engine.metadata import DatasetMetadata
from data_engine.normalization import DataNormalizer
from data_engine.preprocessing import DataPreprocessor
from data_engine.resampling import DataResampler
from data_engine.schema import SchemaValidator
from data_engine.validation import DataValidationReport, DataValidator
from data_engine.versioning import DataVersionManager

logger = get_data_engine_logger("DataEngine")


class DataEngine:
    """Master Institutional Data Engineering Platform Engine for QuantLab."""

    def __init__(self, storage_dir: str = "data_storage") -> None:
        """Initialize DataEngine.

        Args:
            storage_dir: Storage directory path.
        """
        self.data_manager = DataManager(storage_dir=storage_dir)
        self.sources_registry = DataSourceRegistry()
        self.ingestion_engine = DataIngestionEngine()
        self.preprocessor = DataPreprocessor()
        self.version_manager = DataVersionManager()
        self.cache = DataCache()

    def load_dataset_file(self, filepath: str, name: str = "LoadedDataset") -> Tuple[pd.DataFrame, DatasetMetadata]:
        """Load dataset from file, validate schema, and register in data manager.

        Returns:
            Tuple of (DataFrame, DatasetMetadata).
        """
        if filepath.endswith(".parquet") or filepath.endswith(".pq"):
            source = ParquetDataSource(filepath)
        else:
            source = CSVDataSource(filepath)

        df = source.fetch_data()
        df = SchemaValidator.enforce_ohlcv_types(df)

        meta = self.data_manager.register_dataset(name, df)
        self.version_manager.create_snapshot(name, df)
        logger.log_ingestion(name, source.source_type, len(df))
        return df, meta

    def clean_and_normalize(
        self, df: pd.DataFrame, method: str = "zscore", numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Execute missing value imputation, outlier clipping, and scaling normalization.

        Returns:
            Cleaned and normalized DataFrame.
        """
        proc = DataPreprocessor(normalize_method=method)
        return proc.preprocess(df, numeric_columns=numeric_columns)

    def resample(self, df: pd.DataFrame, target_timeframe: str = "1h") -> pd.DataFrame:
        """Resample DataFrame to target timeframe (1s, 5m, 1h, 1D, etc.) using OHLCV rules.

        Returns:
            Resampled DataFrame.
        """
        res = DataResampler.resample_ohlcv(df, target_timeframe=target_timeframe)
        logger.log_resampling("ResampledData", "raw", target_timeframe, len(res))
        return res

    def label_triple_barrier(
        self, df: pd.DataFrame, pt_ratio: float = 0.01, sl_ratio: float = 0.01, max_holding: int = 10
    ) -> pd.Series:
        """Generate Triple Barrier quantitative ML labels (1 = Take Profit, -1 = Stop Loss, 0 = Timeout).

        Returns:
            Label Series.
        """
        return DataLabeler.label_triple_barrier(df, pt_ratio=pt_ratio, sl_ratio=sl_ratio, max_holding=max_holding)

    def validate_quality(self, df: pd.DataFrame) -> DataValidationReport:
        """Run comprehensive data quality telemetry validation.

        Returns:
            DataValidationReport instance.
        """
        return DataValidator.validate_dataset(df)

    def export(self, df: pd.DataFrame, filepath: str, export_format: str = "csv") -> str:
        """Export DataFrame into target format file (csv, parquet, json, excel, markdown)."""
        fmt = export_format.lower()
        if fmt in ("parquet", "pq"):
            return DataExporter.to_parquet(df, filepath)
        elif fmt == "json":
            return DataExporter.to_json(df, filepath)
        elif fmt in ("excel", "xlsx"):
            return DataExporter.to_excel(df, filepath)
        elif fmt in ("markdown", "md"):
            return DataExporter.to_markdown(df, filepath)
        else:
            return DataExporter.to_csv(df, filepath)
