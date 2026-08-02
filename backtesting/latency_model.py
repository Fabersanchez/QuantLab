"""
QuantLab Institutional Latency Models.

Simulates execution delays including network ping, exchange matching engine latency,
broker routing processing time, and multi-bar execution offsets.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class BaseLatencyModel(ABC):
    """Abstract Base Class for all latency and delay models."""

    @abstractmethod
    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Calculate execution delay.

        Args:
            timestamp: Signal creation timestamp.
            bar_index: Sequential bar index in backtest dataset.

        Returns:
            Tuple of (bar_delay_count, latency_seconds).
        """
        pass


class ExecutionDelayModel(BaseLatencyModel):
    """Fixed or variable bar execution delay (e.g., execute next bar open vs same bar close)."""

    def __init__(self, bar_delay: int = 1) -> None:
        """Initialize execution delay model.

        Args:
            bar_delay: Number of bars to delay execution after signal (default: 1 = Next Bar Open).
        """
        if bar_delay < 0:
            raise ValueError("bar_delay must be non-negative.")
        self._bar_delay = int(bar_delay)

    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Return bar execution delay count."""
        return (self._bar_delay, 0.0)


class NetworkDelayModel(BaseLatencyModel):
    """Network transmission latency simulation in milliseconds (e.g., VPS to exchange ping)."""

    def __init__(self, latency_ms: float = 20.0) -> None:
        """Initialize network delay model.

        Args:
            latency_ms: One-way network latency in milliseconds.
        """
        self._ms = max(0.0, float(latency_ms))

    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Return 0 bar delay and latency in seconds."""
        return (0, self._ms / 1000.0)


class ExchangeDelayModel(BaseLatencyModel):
    """Exchange order book queue matching engine latency in milliseconds."""

    def __init__(self, matching_delay_ms: float = 15.0) -> None:
        """Initialize exchange matching engine delay model.

        Args:
            matching_delay_ms: Matching engine queue delay in milliseconds.
        """
        self._ms = max(0.0, float(matching_delay_ms))

    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Return 0 bar delay and latency in seconds."""
        return (0, self._ms / 1000.0)


class BrokerDelayModel(BaseLatencyModel):
    """Broker server processing and risk management pre-trade check delay."""

    def __init__(self, processing_delay_ms: float = 35.0) -> None:
        """Initialize broker processing delay model.

        Args:
            processing_delay_ms: Server routing delay in milliseconds.
        """
        self._ms = max(0.0, float(processing_delay_ms))

    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Return 0 bar delay and latency in seconds."""
        return (0, self._ms / 1000.0)


class CompositeLatencyModel(BaseLatencyModel):
    """Composite latency model combining network, broker, exchange, and bar delays."""

    def __init__(
        self,
        bar_delay: int = 1,
        network_ms: float = 25.0,
        broker_ms: float = 30.0,
        exchange_ms: float = 15.0,
    ) -> None:
        """Initialize composite latency model.

        Args:
            bar_delay: Number of bars offset.
            network_ms: Network ping ms.
            broker_ms: Broker processing ms.
            exchange_ms: Exchange matching engine ms.
        """
        self._bar_delay = max(0, int(bar_delay))
        self._total_ms = max(0.0, float(network_ms)) + max(0.0, float(broker_ms)) + max(0.0, float(exchange_ms))

    def calculate_delay(self, timestamp: Any, bar_index: int = 0) -> Tuple[int, float]:
        """Return total aggregated bar delay and combined time latency in seconds."""
        return (self._bar_delay, self._total_ms / 1000.0)


class LatencyModelFactory:
    """Factory to instantiate latency models from name or configuration dict."""

    @staticmethod
    def create(model_type: str, **kwargs) -> BaseLatencyModel:
        """Create latency model instance.

        Args:
            model_type: Type identifier ('execution', 'network', 'exchange', 'broker', 'composite').
            kwargs: Constructor keyword arguments.

        Returns:
            Instance of BaseLatencyModel.
        """
        m_type = model_type.lower().strip()
        if m_type in ("execution", "execution_delay"):
            return ExecutionDelayModel(**kwargs)
        elif m_type in ("network", "network_delay"):
            return NetworkDelayModel(**kwargs)
        elif m_type in ("exchange", "exchange_delay"):
            return ExchangeDelayModel(**kwargs)
        elif m_type in ("broker", "broker_delay"):
            return BrokerDelayModel(**kwargs)
        elif m_type == "composite":
            return CompositeLatencyModel(**kwargs)
        else:
            raise ValueError(f"Unknown latency model type '{model_type}'.")
