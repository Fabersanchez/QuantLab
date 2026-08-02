"""
QuantLab Execution Noise & Perturbation Injectors.

Simulates execution noise by injecting random perturbations into Bid-Ask Spread,
Slippage, Commission, Latency, and Bar Volatility during Monte Carlo simulations.
"""

from abc import ABC, abstractmethod
import random
from typing import Any, Dict, Optional


class BaseNoiseInjector(ABC):
    """Abstract Base Class for all noise injectors."""

    @abstractmethod
    def perturb(self, value: float) -> float:
        """Apply noise perturbation to target numerical value.

        Args:
            value: Original base value (e.g. spread, slippage, commission, price).

        Returns:
            Perturbed value.
        """
        pass


class SpreadNoiseInjector(BaseNoiseInjector):
    """Injects random noise into bid-ask spread."""

    def __init__(self, noise_std_pips: float = 0.5, point_value: float = 0.0001) -> None:
        """Initialize SpreadNoiseInjector.

        Args:
            noise_std_pips: Standard deviation of spread noise in pips.
            point_value: Pip size.
        """
        self.noise_std = float(noise_std_pips) * float(point_value)

    def perturb(self, value: float) -> float:
        """Add non-negative Gaussian noise to spread."""
        noise = random.gauss(0.0, self.noise_std)
        return max(0.0, value + noise)


class SlippageNoiseInjector(BaseNoiseInjector):
    """Injects random noise into order slippage."""

    def __init__(self, noise_pct: float = 0.2) -> None:
        """Initialize SlippageNoiseInjector.

        Args:
            noise_pct: Decimal percentage variation (e.g. 0.2 = ±20% noise).
        """
        self.noise_pct = float(noise_pct)

    def perturb(self, value: float) -> float:
        """Multiply base slippage by random factor."""
        mult = max(0.0, random.gauss(1.0, self.noise_pct))
        return value * mult


class CommissionNoiseInjector(BaseNoiseInjector):
    """Injects random noise into commission rates."""

    def __init__(self, noise_pct: float = 0.1) -> None:
        """Initialize CommissionNoiseInjector."""
        self.noise_pct = float(noise_pct)

    def perturb(self, value: float) -> float:
        """Multiply commission by random factor."""
        mult = max(0.0, random.gauss(1.0, self.noise_pct))
        return value * mult


class LatencyNoiseInjector(BaseNoiseInjector):
    """Injects random noise into order execution latency (bar delays)."""

    def __init__(self, max_extra_bars: int = 2) -> None:
        """Initialize LatencyNoiseInjector.

        Args:
            max_extra_bars: Maximum random extra bar delay.
        """
        self.max_extra = max(0, int(max_extra_bars))

    def perturb(self, value: float) -> float:
        """Add random integer bar delay."""
        extra = random.randint(0, self.max_extra)
        return value + float(extra)


class VolatilityNoiseInjector(BaseNoiseInjector):
    """Injects random noise into bar high-low volatility range."""

    def __init__(self, noise_pct: float = 0.15) -> None:
        """Initialize VolatilityNoiseInjector."""
        self.noise_pct = float(noise_pct)

    def perturb(self, value: float) -> float:
        """Scale price range by random factor."""
        mult = max(0.1, random.gauss(1.0, self.noise_pct))
        return value * mult


class CompositeNoiseInjector:
    """Composite noise injector aggregating spread, slippage, commission, latency, and volatility noise."""

    def __init__(
        self,
        spread_injector: Optional[SpreadNoiseInjector] = None,
        slippage_injector: Optional[SlippageNoiseInjector] = None,
        commission_injector: Optional[CommissionNoiseInjector] = None,
        latency_injector: Optional[LatencyNoiseInjector] = None,
        volatility_injector: Optional[VolatilityNoiseInjector] = None,
    ) -> None:
        """Initialize CompositeNoiseInjector."""
        self.spread = spread_injector or SpreadNoiseInjector()
        self.slippage = slippage_injector or SlippageNoiseInjector()
        self.commission = commission_injector or CommissionNoiseInjector()
        self.latency = latency_injector or LatencyNoiseInjector()
        self.volatility = volatility_injector or VolatilityNoiseInjector()

    def apply_bar_noise(self, bar_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise perturbations to a market bar dictionary.

        Args:
            bar_dict: Dict with 'open', 'high', 'low', 'close', 'volume', 'spread'.

        Returns:
            Perturbed bar dictionary.
        """
        out = bar_dict.copy()
        high = float(out.get("high", out.get("close", 100.0)))
        low = float(out.get("low", out.get("close", 100.0)))
        close = float(out.get("close", 100.0))

        bar_range = max(0.0, high - low)
        perturbed_range = self.volatility.perturb(bar_range)

        out["high"] = close + (perturbed_range / 2.0)
        out["low"] = max(1e-4, close - (perturbed_range / 2.0))

        if "spread" in out:
            out["spread"] = self.spread.perturb(float(out["spread"]))

        return out
