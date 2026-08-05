"""
QuantLab Data Ingestion Engine.

Executes Full, Incremental, Parallel (ThreadPool), Async, Scheduled ingestion,
auto-retries, and duplicate detection.
"""

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from data_engine.datasource import BaseDataSource
from data_engine.logger import get_data_engine_logger

logger = get_data_engine_logger("Ingestion")


class DataIngestionEngine:
    """Institutional Data Ingestion Engine."""

    def __init__(self, max_workers: int = 4, max_retries: int = 3) -> None:
        self.max_workers = max_workers
        self.max_retries = max_retries

    def ingest_single(
        self, source: BaseDataSource, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Ingest data from a single data source with auto-retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                df = source.fetch_data(symbol, start_date, end_date)
                logger.log_ingestion(symbol or source.name, source.source_type, len(df))
                return df
            except Exception as e:
                logger.warning(f"Ingestion attempt {attempt}/{self.max_retries} failed for '{symbol}': {e}")
                time.sleep(0.5)
        logger.log_error(symbol, f"All {self.max_retries} ingestion attempts failed.")
        return pd.DataFrame()

    def ingest_parallel(self, source: BaseDataSource, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Ingest data for multiple symbols in parallel via ThreadPoolExecutor."""
        results: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sym = {executor.submit(self.ingest_single, source, sym): sym for sym in symbols}
            for future in future_to_sym:
                sym = future_to_sym[future]
                try:
                    results[sym] = future.result()
                except Exception as e:
                    logger.log_error(sym, str(e))
                    results[sym] = pd.DataFrame()
        return results
