"""
QuantLab Optimization Evaluation Cache.

Prevents redundant evaluations of identical strategy parameter combinations over identical
market datasets using SHA-256 digests. Thread-safe with hit/miss telemetry.
"""

from dataclasses import dataclass, field
import hashlib
import json
import threading
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    """Dataclass holding cached evaluation outputs."""

    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    fitness_score: float
    is_valid: bool
    execution_time_sec: float


class OptimizationCache:
    """Institutional Thread-Safe Optimization Cache."""

    def __init__(self) -> None:
        """Initialize OptimizationCache."""
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def hash_parameters(parameters: Dict[str, Any]) -> str:
        """Compute SHA-256 hash digest of parameter dictionary."""
        raw = json.dumps(parameters, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def hash_dataset(dataset_info: Any) -> str:
        """Compute SHA-256 hash digest of market dataset metadata."""
        if isinstance(dataset_info, dict):
            raw = json.dumps(dataset_info, sort_keys=True, default=str).encode("utf-8")
        elif hasattr(dataset_info, "shape"):
            raw = f"dataset_shape_{dataset_info.shape}".encode("utf-8")
        else:
            raw = str(dataset_info).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def hash_strategy(strategy_cls_or_name: Any) -> str:
        """Compute SHA-256 hash digest of strategy name/class."""
        name = getattr(strategy_cls_or_name, "__name__", str(strategy_cls_or_name))
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def generate_key(
        self, strategy: Any, dataset_info: Any, parameters: Dict[str, Any]
    ) -> str:
        """Generate combined SHA-256 cache key.

        Returns:
            Hexadecimal SHA-256 key string.
        """
        strat_h = self.hash_strategy(strategy)
        ds_h = self.hash_dataset(dataset_info)
        params_h = self.hash_parameters(parameters)
        combined = f"{strat_h}:{ds_h}:{params_h}".encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Fetch cached evaluation entry by key."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(
        self,
        key: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        fitness_score: float,
        is_valid: bool,
        execution_time_sec: float,
    ) -> None:
        """Store evaluation output in cache."""
        with self._lock:
            entry = CacheEntry(
                parameters=parameters,
                metrics=metrics,
                fitness_score=fitness_score,
                is_valid=is_valid,
                execution_time_sec=execution_time_sec,
            )
            self._cache[key] = entry

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear cache storage."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get cache telemetry metrics (total items, hits, misses, hit ratio)."""
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": float(ratio),
            }
