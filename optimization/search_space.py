"""
QuantLab Strategy Search Space Manager.

Constructs flat and hierarchical parameter search spaces, manages sampling, grid generation,
vectorized normalization/denormalization, and domain validation.
"""

import itertools
from typing import Any, Dict, List, Generator, Optional, Union
import numpy as np

from optimization.parameter_space import (
    BooleanParameter,
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    Parameter,
)


class SearchSpace:
    """Institutional Strategy Search Space definition container."""

    def __init__(self, name: str = "DefaultSearchSpace") -> None:
        """Initialize SearchSpace.

        Args:
            name: Search space name.
        """
        self.name = name
        self.parameters: Dict[str, Parameter] = {}
        self.groups: Dict[str, "SearchSpace"] = {}

    def add_parameter(self, param: Parameter) -> "SearchSpace":
        """Add a parameter to search space.

        Args:
            param: Parameter instance.

        Returns:
            self for method chaining.
        """
        self.parameters[param.name] = param
        return self

    def add_group(self, group_name: str, sub_space: "SearchSpace") -> "SearchSpace":
        """Add a hierarchical sub-search space group.

        Args:
            group_name: Sub-space group identifier.
            sub_space: Child SearchSpace instance.

        Returns:
            self for method chaining.
        """
        self.groups[group_name] = sub_space
        return self

    @property
    def flat_parameters(self) -> Dict[str, Parameter]:
        """Flatten all parameters including sub-groups into dot-notation dictionary."""
        flat: Dict[str, Parameter] = {}
        for p_name, param in self.parameters.items():
            flat[p_name] = param
        for g_name, group in self.groups.items():
            for sub_p_name, sub_param in group.flat_parameters.items():
                flat[f"{g_name}.{sub_p_name}"] = sub_param
        return flat

    @property
    def dimension(self) -> int:
        """Total degrees of freedom (number of parameters)."""
        return len(self.flat_parameters)

    def sample(self) -> Dict[str, Any]:
        """Sample a single random parameter combination from search space."""
        sampled: Dict[str, Any] = {}
        for p_name, param in self.parameters.items():
            sampled[p_name] = param.sample()
        for g_name, group in self.groups.items():
            sampled[g_name] = group.sample()
        return sampled

    def sample_batch(self, n: int) -> List[Dict[str, Any]]:
        """Sample n random parameter combinations.

        Args:
            n: Number of samples.

        Returns:
            List of parameter dictionaries.
        """
        return [self.sample() for _ in range(n)]

    def grid_points(self, points_per_dim: int = 5) -> Generator[Dict[str, Any], None, None]:
        """Generate grid points over discrete/continuous parameter spaces.

        Args:
            points_per_dim: Number of candidate points per numeric dimension.

        Yields:
            Parameter dictionary instances.
        """
        flat = self.flat_parameters
        param_names = list(flat.keys())
        dim_values: List[List[Any]] = []

        for p_name in param_names:
            param = flat[p_name]
            if isinstance(param, BooleanParameter):
                dim_values.append([True, False])
            elif isinstance(param, CategoricalParameter):
                dim_values.append(list(param.choices))
            elif isinstance(param, IntegerParameter):
                pts = np.linspace(param.min_val, param.max_val, min(points_per_dim, param.max_val - param.min_val + 1))
                unique_ints = sorted(list(set([int(round(x)) for x in pts])))
                dim_values.append(unique_ints)
            elif isinstance(param, FloatParameter):
                pts = np.linspace(param.min_val, param.max_val, points_per_dim)
                dim_values.append([float(x) for x in pts])
            else:
                dim_values.append([param.sample() for _ in range(points_per_dim)])

        for combo in itertools.product(*dim_values):
            point_dict: Dict[str, Any] = {}
            for idx, p_name in enumerate(param_names):
                point_dict[p_name] = combo[idx]
            yield point_dict

    def contains(self, param_dict: Dict[str, Any]) -> bool:
        """Check if parameter dictionary belongs to and satisfies search space constraints.

        Args:
            param_dict: Dictionary of parameter values.

        Returns:
            True if valid, False otherwise.
        """
        flat = self.flat_parameters
        for p_name, param in flat.items():
            if p_name not in param_dict:
                return False
            if not param.validate(param_dict[p_name]):
                return False
        return True

    def normalize(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """Map parameter dictionary into normalized 1D NumPy vector [0.0, 1.0]^D."""
        flat = self.flat_parameters
        vec = []
        for p_name in flat.keys():
            param = flat[p_name]
            val = param_dict.get(p_name, param.default)
            vec.append(param.normalize(val))
        return np.array(vec, dtype=np.float64)

    def denormalize(self, vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Map normalized 1D float vector [0.0, 1.0]^D back to parameter dictionary."""
        flat = self.flat_parameters
        param_names = list(flat.keys())
        result: Dict[str, Any] = {}

        for idx, p_name in enumerate(param_names):
            param = flat[p_name]
            norm_val = float(vector[idx]) if idx < len(vector) else 0.5
            result[p_name] = param.denormalize(norm_val)

        return result
