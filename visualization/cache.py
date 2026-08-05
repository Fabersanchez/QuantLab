"""
QuantLab Intelligent Chart Rendering Cache.

Prevents redundant figure generation by storing rendered chart outputs keyed by SHA-256
digests of input datasets, chart parameters, themes, and dimensions.
"""

from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Any, Dict, Optional


@dataclass
class CachedChart:
    """Dataclass holding cached figure binary or base64 data."""

    key: str
    chart_type: str
    image_bytes: bytes
    render_time_ms: float


class VisualizationCache:
    """Institutional Thread-Safe Visualization Cache."""

    def __init__(self) -> None:
        """Initialize VisualizationCache."""
        self._cache: Dict[str, CachedChart] = {}
        self._lock = threading.RLock()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def generate_key(
        chart_type: str, data_repr: Any, parameters: Dict[str, Any], theme_name: str
    ) -> str:
        """Compute SHA-256 cache digest key.

        Returns:
            Hexadecimal SHA-256 key string.
        """
        if isinstance(data_repr, dict):
            d_str = json.dumps(data_repr, sort_keys=True, default=str)
        elif hasattr(data_repr, "to_csv"):
            d_str = str(data_repr.shape) + str(data_repr.tail(5).values)
        else:
            d_str = str(data_repr)

        p_str = json.dumps(parameters, sort_keys=True, default=str)
        raw = f"{chart_type}:{d_str}:{p_str}:{theme_name}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> Optional[CachedChart]:
        """Fetch cached chart by key."""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(
        self, key: str, chart_type: str, image_bytes: bytes, render_time_ms: float
    ) -> None:
        """Store rendered chart in cache."""
        with self._lock:
            self._cache[key] = CachedChart(
                key=key,
                chart_type=chart_type,
                image_bytes=image_bytes,
                render_time_ms=render_time_ms,
            )

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
        """Get telemetry statistics."""
        with self._lock:
            total = self._hits + self._misses
            ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": float(ratio),
            }
