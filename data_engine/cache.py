"""
QuantLab Thread-Safe Data Processing Cache.

Provides intelligent thread-safe caching for processed DataFrames.
"""

import threading
from typing import Any, Dict, Optional
import pandas as pd


class DataCache:
    """Institutional Thread-Safe Data Processing Cache."""

    def __init__(self) -> None:
        self._cache: Dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Fetch cached DataFrame by key."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key].copy()
            self._misses += 1
            return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Store DataFrame in cache."""
        with self._lock:
            self._cache[key] = df.copy()

    def clear(self) -> None:
        """Clear cache storage."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": float(ratio),
            }
