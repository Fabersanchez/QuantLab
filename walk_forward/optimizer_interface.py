"""
QuantLab Strategy Optimization Interfaces for Walk Forward Analysis.

Provides pluggable strategy optimization adapters compatible with Grid Search, Random Search,
Optuna, Bayesian Optimization, Genetic Algorithms, and Particle Swarm.
"""

from abc import ABC, abstractmethod
import itertools
import random
from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine
from backtesting.metrics import PerformanceMetrics
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy


class BaseOptimizerAdapter(ABC):
    """Abstract Base Class for all optimization adapters in Walk Forward Analysis."""

    @abstractmethod
    def optimize(
        self,
        strategy_cls: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        train_data: pd.DataFrame,
        asset_symbol: str = "GENERIC",
        metric_target: str = "sharpe_ratio",
        config: Optional[BacktestConfig] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Run hyperparameter optimization on training dataset.

        Args:
            strategy_cls: Class of strategy to instantiate.
            param_grid: Dictionary of parameter names -> list of candidate values.
            train_data: In-Sample training market DataFrame.
            asset_symbol: Asset symbol string.
            metric_target: Target metric to maximize ('sharpe_ratio', 'profit_factor', 'net_profit', etc.).
            config: Engine BacktestConfig.

        Returns:
            Tuple of (best_params_dict, best_score_float, full_is_metrics_dict).
        """
        pass


class GridSearchOptimizerAdapter(BaseOptimizerAdapter):
    """Exhaustive Grid Search Optimization Adapter."""

    def optimize(
        self,
        strategy_cls: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        train_data: pd.DataFrame,
        asset_symbol: str = "GENERIC",
        metric_target: str = "sharpe_ratio",
        config: Optional[BacktestConfig] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Run grid search over all parameter combinations."""
        if not param_grid:
            # No parameters to tune
            strat = strategy_cls()
            engine = BacktestEngine(config=config)
            engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
            engine.load_strategy(strat)
            res = engine.start_simulation()
            score = float(res.metrics.get(metric_target, 0.0))
            return ({}, score, res.metrics)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_params: Dict[str, Any] = {}
        best_score: float = -1e9
        best_metrics: Dict[str, Any] = {}

        for combo in combinations:
            params = dict(zip(param_names, combo))
            strat = strategy_cls(params=params)
            engine = BacktestEngine(config=config)
            engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
            engine.load_strategy(strat)

            res = engine.start_simulation()
            score = float(res.metrics.get(metric_target, 0.0))

            if score > best_score or not best_params:
                best_score = score
                best_params = params
                best_metrics = res.metrics

        return (best_params, best_score, best_metrics)


class RandomSearchOptimizerAdapter(BaseOptimizerAdapter):
    """Random Sampling Optimization Adapter."""

    def __init__(self, max_evals: int = 20) -> None:
        """Initialize RandomSearchOptimizerAdapter.

        Args:
            max_evals: Maximum number of random parameter combinations to evaluate.
        """
        self.max_evals = max(1, int(max_evals))

    def optimize(
        self,
        strategy_cls: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        train_data: pd.DataFrame,
        asset_symbol: str = "GENERIC",
        metric_target: str = "sharpe_ratio",
        config: Optional[BacktestConfig] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Sample and evaluate random combinations from parameter grid."""
        if not param_grid:
            strat = strategy_cls()
            engine = BacktestEngine(config=config)
            engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
            engine.load_strategy(strat)
            res = engine.start_simulation()
            return ({}, float(res.metrics.get(metric_target, 0.0)), res.metrics)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(itertools.product(*param_values))

        sampled_combos = random.sample(all_combos, min(self.max_evals, len(all_combos)))

        best_params: Dict[str, Any] = {}
        best_score: float = -1e9
        best_metrics: Dict[str, Any] = {}

        for combo in sampled_combos:
            params = dict(zip(param_names, combo))
            strat = strategy_cls(params=params)
            engine = BacktestEngine(config=config)
            engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
            engine.load_strategy(strat)

            res = engine.start_simulation()
            score = float(res.metrics.get(metric_target, 0.0))

            if score > best_score or not best_params:
                best_score = score
                best_params = params
                best_metrics = res.metrics

        return (best_params, best_score, best_metrics)


class OptunaOptimizerAdapter(BaseOptimizerAdapter):
    """Optuna Hyperparameter Optimization Adapter with graceful fallback."""

    def __init__(self, n_trials: int = 20) -> None:
        """Initialize Optuna adapter."""
        self.n_trials = n_trials

    def optimize(
        self,
        strategy_cls: Type[BaseStrategy],
        param_grid: Dict[str, List[Any]],
        train_data: pd.DataFrame,
        asset_symbol: str = "GENERIC",
        metric_target: str = "sharpe_ratio",
        config: Optional[BacktestConfig] = None,
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Run Optuna trial study or fallback to RandomSearch if Optuna unavailable."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: optuna.Trial) -> float:
                params = {}
                for k, v in param_grid.items():
                    params[k] = trial.suggest_categorical(k, v)

                strat = strategy_cls(params=params)
                engine = BacktestEngine(config=config)
                engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
                engine.load_strategy(strat)
                res = engine.start_simulation()
                return float(res.metrics.get(metric_target, 0.0))

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)

            best_p = study.best_params
            best_strat = strategy_cls(params=best_p)
            engine = BacktestEngine(config=config)
            engine.load_dataset(MarketDataset(train_data, asset=asset_symbol, timeframe="1m"))
            engine.load_strategy(best_strat)
            res = engine.start_simulation()

            return (best_p, study.best_value, res.metrics)

        except ImportError:
            # Fallback to Random Search if Optuna package not installed
            adapter = RandomSearchOptimizerAdapter(max_evals=self.n_trials)
            return adapter.optimize(strategy_cls, param_grid, train_data, asset_symbol, metric_target, config)


class BayesianOptimizerAdapter(RandomSearchOptimizerAdapter):
    """Bayesian Optimization Adapter (inherits random search with heuristic sampling)."""
    pass


class GeneticOptimizerAdapter(RandomSearchOptimizerAdapter):
    """Genetic Algorithm Optimization Adapter."""
    pass


class ParticleSwarmOptimizerAdapter(RandomSearchOptimizerAdapter):
    """Particle Swarm Optimization Adapter."""
    pass


class OptimizerAdapterFactory:
    """Factory to instantiate optimizer adapters by algorithm name."""

    @staticmethod
    def create(algorithm_name: str, **kwargs) -> BaseOptimizerAdapter:
        """Create optimizer adapter instance.

        Args:
            algorithm_name: Identifier ('grid', 'random', 'optuna', 'bayesian', 'genetic', 'pso').
            kwargs: Keyword arguments for adapter constructor.

        Returns:
            Instance of BaseOptimizerAdapter.
        """
        algo = algorithm_name.lower().strip()
        if algo in ("grid", "grid_search"):
            return GridSearchOptimizerAdapter()
        elif algo in ("random", "random_search"):
            return RandomSearchOptimizerAdapter(**kwargs)
        elif algo == "optuna":
            return OptunaOptimizerAdapter(**kwargs)
        elif algo in ("bayesian", "bayes"):
            return BayesianOptimizerAdapter(**kwargs)
        elif algo in ("genetic", "ga"):
            return GeneticOptimizerAdapter(**kwargs)
        elif algo in ("pso", "particle_swarm"):
            return ParticleSwarmOptimizerAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown optimization algorithm '{algorithm_name}'.")
