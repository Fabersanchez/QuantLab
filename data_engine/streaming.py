"""
QuantLab Real-Time Data Streaming Buffer.

Buffers real-time streaming Ticks, OHLC bars, Order Book Depth (Level II), Volume, and Trade executions.
"""

from collections import deque
import threading
from typing import Any, Dict, List, Optional
import pandas as pd


class DataStreamer:
    """Institutional Real-Time Data Streamer & Buffer."""

    def __init__(self, max_buffer_size: int = 10000) -> None:
        self.max_buffer_size = max_buffer_size
        self._ticks: deque = deque(maxlen=max_buffer_size)
        self._bars: deque = deque(maxlen=max_buffer_size)
        self._lock = threading.RLock()

    def push_tick(self, timestamp: Any, symbol: str, price: float, volume: float, side: str = "BUY") -> None:
        """Push real-time tick record to buffer."""
        with self._lock:
            self._ticks.append(
                {"timestamp": timestamp, "symbol": symbol, "price": price, "volume": volume, "side": side}
            )

    def push_bar(self, timestamp: Any, symbol: str, open_p: float, high_p: float, low_p: float, close_p: float, vol: float) -> None:
        """Push real-time OHLC bar record to buffer."""
        with self._lock:
            self._bars.append(
                {"timestamp": timestamp, "symbol": symbol, "open": open_p, "high": high_p, "low": low_p, "close": close_p, "volume": vol}
            )

    def get_ticks_df(self) -> pd.DataFrame:
        """Fetch accumulated ticks buffer as DataFrame."""
        with self._lock:
            return pd.DataFrame(list(self._ticks))

    def get_bars_df(self) -> pd.DataFrame:
        """Fetch accumulated bars buffer as DataFrame."""
        with self._lock:
            return pd.DataFrame(list(self._bars))
