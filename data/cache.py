"""
QuantLab In-Memory Cache System.

Provides high-performance in-memory caching with Least Recently Used (LRU)
eviction policy and Time-To-Live (TTL) expiration support for market datasets.
"""

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional


class CacheEntry:
    """Individual entry container in memory cache."""

    def __init__(self, value: Any, ttl_seconds: Optional[float] = None) -> None:
        self.value = value
        self.created_at = datetime.now(timezone.utc)
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


class MemoryCache:
    """LRU in-memory cache with TTL support."""

    def __init__(
        self, max_size: int = 100, default_ttl_seconds: Optional[float] = None
    ) -> None:
        """Initialize MemoryCache.

        Args:
            max_size: Maximum capacity before LRU eviction.
            default_ttl_seconds: Optional default TTL in seconds.
        """
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key, updating LRU order if valid.

        Args:
            key: Unique key identifier string.

        Returns:
            Cached value if found and unexpired, None otherwise.
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            return None

        self._cache.move_to_end(key)
        return entry.value

    def set(
        self, key: str, value: Any, ttl_seconds: Optional[float] = None
    ) -> None:
        """Insert or update key-value pair in cache.

        Args:
            key: Unique key identifier string.
            value: Data item to store.
            ttl_seconds: Custom TTL override in seconds.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._max_size:
            # Evict LRU item (first item in OrderedDict)
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value, ttl_seconds=ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache.

        Returns:
            True if key was removed, False if key was not found.
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries and return count of evicted items."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    def size(self) -> int:
        """Return current number of unexpired entries."""
        self.cleanup_expired()
        return len(self._cache)
