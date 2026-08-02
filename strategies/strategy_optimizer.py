"""
QuantLab Strategy Optimizer Architecture.

Provides abstract optimizer interfaces for parameter hyperparameter tuning
(Grid Search, Random Search, Bayesian Optimization, Genetic Algorithms, Optuna, Hyperopt).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class OptimizationResult:
    """Container for strategy optimization results."""

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    all_trials: List[Dict[str, Any]] = field(default_factory=list)


class BaseOptimizer(ABC):
    """Abstract Base Class for strategy hyperparameter optimizers."""

    @abstractmethod
    def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        """Run hyperparameter search and return OptimizationResult."""
        pass


class GridSearchOptimizer(BaseOptimizer):
    """Grid Search parameter optimizer stub interface."""

    def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        return OptimizationResult(best_params={}, best_score=0.0)


class RandomSearchOptimizer(BaseOptimizer):
    """Random Search parameter optimizer stub interface."""

    def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        return OptimizationResult(best_params={}, best_score=0.0)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization stub interface (Optuna / Hyperopt integration ready)."""

    def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        return OptimizationResult(best_params={}, best_score=0.0)


class GeneticOptimizer(BaseOptimizer):
    """Genetic Algorithm parameter optimizer stub interface."""

    def optimize(
        self,
        strategy_cls: Any,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
    ) -> OptimizationResult:
        return OptimizationResult(best_params={}, best_score=0.0)


class OptimizerFactory:
    """Factory for acquiring strategy optimizer instances."""

    _MAP = {
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
        "bayesian": BayesianOptimizer,
        "genetic": GeneticOptimizer,
    }

    @classmethod
    def get_optimizer(cls, name: str) -> BaseOptimizer:
        key = name.lower()
        if key not in cls._MAP:
            raise ValueError(f"Unsupported optimizer algorithm: {name}")
        return cls._MAP[key]()
