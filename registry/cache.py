"""
QuantLab Governance Registry In-Memory Cache.

Provides thread-safe in-memory caching for active governance records and lineage nodes.
"""

import threading
from typing import Any, Dict, Optional


class RegistryCache:
    """Institutional Thread-Safe Registry Cache."""

    def __init__(self) -> None:
        """Initialize RegistryCache."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Fetch cached payload by key."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        """Store payload in cache."""
        with self._lock:
            self._cache[key] = dict(payload)

    def remove(self, key: str) -> None:
        """Remove key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """Clear all cached records."""
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
