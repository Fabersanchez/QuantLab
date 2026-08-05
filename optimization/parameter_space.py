"""
QuantLab Strategy Parameter Space System.

Provides professional type specifications for strategy parameters: Integer, Float, Boolean,
Categorical, Discrete, Continuous, Log Scale, Uniform, Normal, and Custom types.
Supports validation, sampling, normalization [0, 1], and denormalization.
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np


class Parameter(ABC):
    """Abstract Base Class for strategy optimization parameters."""

    def __init__(self, name: str, default: Optional[Any] = None) -> None:
        """Initialize Parameter base class.

        Args:
            name: Parameter unique identifier.
            default: Optional default parameter value.
        """
        self.name = name
        self.default = default

    @abstractmethod
    def sample(self) -> Any:
        """Randomly sample a valid parameter value."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if value satisfies parameter domain constraints."""
        pass

    @abstractmethod
    def normalize(self, value: Any) -> float:
        """Map parameter value to normalized continuous float range [0.0, 1.0]."""
        pass

    @abstractmethod
    def denormalize(self, norm_value: float) -> Any:
        """Map normalized float [0.0, 1.0] back to parameter domain value."""
        pass


class IntegerParameter(Parameter):
    """Integer parameter bounded by min_val and max_val with optional step."""

    def __init__(self, name: str, min_val: int, max_val: int, step: int = 1, default: Optional[int] = None) -> None:
        super().__init__(name, default if default is not None else min_val)
        if min_val > max_val:
            raise ValueError(f"IntegerParameter '{name}': min_val ({min_val}) cannot exceed max_val ({max_val}).")
        self.min_val = min_val
        self.max_val = max_val
        self.step = max(1, step)

    def sample(self) -> int:
        choices = list(range(self.min_val, self.max_val + 1, self.step))
        return random.choice(choices)

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, np.integer)):
            return False
        return self.min_val <= value <= self.max_val

    def normalize(self, value: int) -> float:
        val = max(self.min_val, min(self.max_val, int(value)))
        if self.max_val == self.min_val:
            return 0.5
        return float((val - self.min_val) / (self.max_val - self.min_val))

    def denormalize(self, norm_value: float) -> int:
        clamped = max(0.0, min(1.0, float(norm_value)))
        raw = self.min_val + clamped * (self.max_val - self.min_val)
        stepped = round((raw - self.min_val) / self.step) * self.step + self.min_val
        return int(max(self.min_val, min(self.max_val, stepped)))


class FloatParameter(Parameter):
    """Float parameter bounded by min_val and max_val with continuous or stepped resolution."""

    def __init__(
        self,
        name: str,
        min_val: float,
        max_val: float,
        step: Optional[float] = None,
        default: Optional[float] = None,
    ) -> None:
        super().__init__(name, default if default is not None else min_val)
        if min_val > max_val:
            raise ValueError(f"FloatParameter '{name}': min_val ({min_val}) cannot exceed max_val ({max_val}).")
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.step = float(step) if step is not None else None

    def sample(self) -> float:
        val = random.uniform(self.min_val, self.max_val)
        if self.step:
            val = round((val - self.min_val) / self.step) * self.step + self.min_val
        return float(max(self.min_val, min(self.max_val, val)))

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float, np.floating, np.integer)):
            return False
        return self.min_val <= float(value) <= self.max_val

    def normalize(self, value: float) -> float:
        val = max(self.min_val, min(self.max_val, float(value)))
        if self.max_val == self.min_val:
            return 0.5
        return float((val - self.min_val) / (self.max_val - self.min_val))

    def denormalize(self, norm_value: float) -> float:
        clamped = max(0.0, min(1.0, float(norm_value)))
        raw = self.min_val + clamped * (self.max_val - self.min_val)
        if self.step:
            raw = round((raw - self.min_val) / self.step) * self.step + self.min_val
        return float(max(self.min_val, min(self.max_val, raw)))


class ContinuousParameter(FloatParameter):
    """Alias for Continuous Float parameter."""

    pass


class BooleanParameter(Parameter):
    """Boolean parameter (True / False)."""

    def __init__(self, name: str, default: Optional[bool] = None) -> None:
        super().__init__(name, default if default is not None else False)

    def sample(self) -> bool:
        return random.choice([True, False])

    def validate(self, value: Any) -> bool:
        return isinstance(value, (bool, np.bool_))

    def normalize(self, value: bool) -> float:
        return 1.0 if bool(value) else 0.0

    def denormalize(self, norm_value: float) -> bool:
        return float(norm_value) >= 0.5


class CategoricalParameter(Parameter):
    """Categorical parameter selecting from a discrete set of choices."""

    def __init__(self, name: str, choices: List[Any], default: Optional[Any] = None) -> None:
        if not choices:
            raise ValueError(f"CategoricalParameter '{name}' requires non-empty choices list.")
        super().__init__(name, default if default is not None else choices[0])
        self.choices = choices

    def sample(self) -> Any:
        return random.choice(self.choices)

    def validate(self, value: Any) -> bool:
        return value in self.choices

    def normalize(self, value: Any) -> float:
        if value not in self.choices:
            return 0.0
        idx = self.choices.index(value)
        if len(self.choices) == 1:
            return 0.5
        return float(idx / (len(self.choices) - 1))

    def denormalize(self, norm_value: float) -> Any:
        clamped = max(0.0, min(1.0, float(norm_value)))
        idx = int(round(clamped * (len(self.choices) - 1)))
        return self.choices[min(len(self.choices) - 1, max(0, idx))]


class DiscreteParameter(CategoricalParameter):
    """Alias for Categorical/Discrete values set parameter."""

    pass


class LogScaleParameter(Parameter):
    """Logarithmic scale float parameter between min_val and max_val (> 0)."""

    def __init__(self, name: str, min_val: float, max_val: float, default: Optional[float] = None) -> None:
        if min_val <= 0 or max_val <= 0:
            raise ValueError(f"LogScaleParameter '{name}' requires strictly positive bounds (> 0).")
        super().__init__(name, default if default is not None else min_val)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.log_min = math.log(self.min_val)
        self.log_max = math.log(self.max_val)

    def sample(self) -> float:
        log_val = random.uniform(self.log_min, self.log_max)
        return float(math.exp(log_val))

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float, np.floating, np.integer)):
            return False
        return self.min_val <= float(value) <= self.max_val

    def normalize(self, value: float) -> float:
        val = max(self.min_val, min(self.max_val, float(value)))
        log_v = math.log(val)
        return float((log_v - self.log_min) / (self.log_max - self.log_min))

    def denormalize(self, norm_value: float) -> float:
        clamped = max(0.0, min(1.0, float(norm_value)))
        log_v = self.log_min + clamped * (self.log_max - self.log_min)
        return float(math.exp(log_v))


class UniformParameter(FloatParameter):
    """Uniformly distributed float parameter."""

    pass


class NormalParameter(Parameter):
    """Gaussian / Normal distribution parameter with mean, std, min_val, and max_val."""

    def __init__(
        self,
        name: str,
        mean: float,
        std: float,
        min_val: float,
        max_val: float,
        default: Optional[float] = None,
    ) -> None:
        super().__init__(name, default if default is not None else mean)
        self.mean = float(mean)
        self.std = float(std)
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def sample(self) -> float:
        val = random.gauss(self.mean, self.std)
        return float(max(self.min_val, min(self.max_val, val)))

    def validate(self, value: Any) -> bool:
        if not isinstance(value, (int, float, np.floating, np.integer)):
            return False
        return self.min_val <= float(value) <= self.max_val

    def normalize(self, value: float) -> float:
        val = max(self.min_val, min(self.max_val, float(value)))
        return float((val - self.min_val) / (self.max_val - self.min_val))

    def denormalize(self, norm_value: float) -> float:
        clamped = max(0.0, min(1.0, float(norm_value)))
        return float(self.min_val + clamped * (self.max_val - self.min_val))


class CustomParameter(Parameter):
    """Custom parameter with user-defined sampler and validator functions."""

    def __init__(
        self,
        name: str,
        sampler_fn: Callable[[], Any],
        validator_fn: Callable[[Any], bool],
        normalizer_fn: Callable[[Any], float],
        denormalizer_fn: Callable[[float], Any],
        default: Optional[Any] = None,
    ) -> None:
        super().__init__(name, default)
        self.sampler_fn = sampler_fn
        self.validator_fn = validator_fn
        self.normalizer_fn = normalizer_fn
        self.denormalizer_fn = denormalizer_fn

    def sample(self) -> Any:
        return self.sampler_fn()

    def validate(self, value: Any) -> bool:
        return self.validator_fn(value)

    def normalize(self, value: Any) -> float:
        return self.normalizer_fn(value)

    def denormalize(self, norm_value: float) -> Any:
        return self.denormalizer_fn(norm_value)
