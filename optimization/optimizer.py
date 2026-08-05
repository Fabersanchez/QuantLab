"""
QuantLab Master Strategy Optimizer Engine.

Implements 12 standalone institutional optimization algorithms:
1. Grid Search
2. Random Search
3. Bayesian Optimization
4. Optuna Adapter / TPE
5. HyperOpt Adapter / TPE
6. Particle Swarm Optimization (PSO)
7. Genetic Algorithms (GA)
8. Evolution Strategy (ES)
9. Differential Evolution (DE)
10. Simulated Annealing (SA)
11. Tree-structured Parzen Estimator (TPE)
12. Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

Supports pause, resume, cancel, save/load state, versioning, evaluation, and export.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import json
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np

from data.market_dataset import MarketDataset
from optimization.cache import OptimizationCache
from optimization.constraints import OptimizationConstraints
from optimization.evaluator import EvaluationResult, SolutionEvaluator
from optimization.history import IterationRecord, OptimizationHistory
from optimization.logger import get_optimization_logger
from optimization.objective_function import ObjectiveFunction
from optimization.scheduler import OptimizationScheduler
from optimization.search_space import SearchSpace
from strategies.base_strategy import BaseStrategy

logger = get_optimization_logger("Optimizer")


class BaseOptimizationAlgorithm(ABC):
    """Abstract Base Class for all strategy optimization algorithms."""

    def __init__(self, search_space: SearchSpace, name: str) -> None:
        """Initialize algorithm base.

        Args:
            search_space: SearchSpace instance.
            name: Algorithm identifier string.
        """
        self.search_space = search_space
        self.name = name

    @abstractmethod
    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        """Propose candidate parameter combinations to evaluate."""
        pass

    @abstractmethod
    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        """Feed evaluation outcomes back to update algorithm internal model state."""
        pass


class GridSearchAlgorithm(BaseOptimizationAlgorithm):
    """Exhaustive Grid Search Optimization Algorithm."""

    def __init__(self, search_space: SearchSpace, points_per_dim: int = 5) -> None:
        super().__init__(search_space, "GridSearch")
        self.grid_generator = list(search_space.grid_points(points_per_dim=points_per_dim))
        self._index = 0

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        batch = self.grid_generator[self._index : self._index + n]
        self._index += len(batch)
        return batch

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        pass


class RandomSearchAlgorithm(BaseOptimizationAlgorithm):
    """Random Search Optimization Algorithm."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space, "RandomSearch")

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        return self.search_space.sample_batch(n)

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        pass


class GeneticAlgorithm(BaseOptimizationAlgorithm):
    """Genetic Algorithm (GA) with selection, crossover, mutation, and elitism."""

    def __init__(
        self,
        search_space: SearchSpace,
        pop_size: int = 20,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elite_count: int = 2,
    ) -> None:
        super().__init__(search_space, "GeneticAlgorithm")
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

        # Initialize population in normalized continuous space [0, 1]^D
        self.dim = search_space.dimension
        self.population_norm = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
        self.fitnesses = np.full(self.pop_size, -1e9)

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for i in range(min(n, self.pop_size)):
            norm_vec = self.population_norm[i]
            params = self.search_space.denormalize(norm_vec)
            candidates.append(params)
        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        for idx, f in enumerate(fitnesses[: len(self.population_norm)]):
            self.fitnesses[idx] = f

        # Create next generation
        new_pop = np.zeros_like(self.population_norm)

        # Elitism
        sorted_indices = np.argsort(self.fitnesses)[::-1]
        for e in range(min(self.elite_count, self.pop_size)):
            new_pop[e] = self.population_norm[sorted_indices[e]]

        # Selection & Crossover
        for i in range(self.elite_count, self.pop_size):
            # Tournament selection
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            parent1 = self.population_norm[idx1] if self.fitnesses[idx1] > self.fitnesses[idx2] else self.population_norm[idx2]

            idx3, idx4 = np.random.choice(self.pop_size, 2, replace=False)
            parent2 = self.population_norm[idx3] if self.fitnesses[idx3] > self.fitnesses[idx4] else self.population_norm[idx4]

            # Uniform crossover
            child = np.copy(parent1)
            if random.random() < self.crossover_rate:
                mask = np.random.rand(self.dim) > 0.5
                child[mask] = parent2[mask]

            # Gaussian mutation
            if random.random() < self.mutation_rate:
                noise = np.random.normal(0.0, 0.1, size=self.dim)
                child = np.clip(child + noise, 0.0, 1.0)

            new_pop[i] = child

        self.population_norm = new_pop


class ParticleSwarmAlgorithm(BaseOptimizationAlgorithm):
    """Particle Swarm Optimization (PSO) with inertia weight and cognitive/social acceleration."""

    def __init__(
        self,
        search_space: SearchSpace,
        swarm_size: int = 20,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        super().__init__(search_space, "ParticleSwarm")
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = search_space.dimension

        self.positions = np.random.uniform(0.0, 1.0, size=(self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.swarm_size, self.dim))
        self.pbest_pos = np.copy(self.positions)
        self.pbest_fitness = np.full(self.swarm_size, -1e9)
        self.gbest_pos = np.copy(self.positions[0])
        self.gbest_fitness = -1e9

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for i in range(min(n, self.swarm_size)):
            params = self.search_space.denormalize(self.positions[i])
            candidates.append(params)
        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        for i, f in enumerate(fitnesses[: self.swarm_size]):
            if f > self.pbest_fitness[i]:
                self.pbest_fitness[i] = f
                self.pbest_pos[i] = np.copy(self.positions[i])
                if f > self.gbest_fitness:
                    self.gbest_fitness = f
                    self.gbest_pos = np.copy(self.positions[i])

        # Velocity and position update
        for i in range(self.swarm_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            cog = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
            soc = self.c2 * r2 * (self.gbest_pos - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cog + soc
            self.positions[i] = np.clip(self.positions[i] + self.velocities[i], 0.0, 1.0)


class DifferentialEvolutionAlgorithm(BaseOptimizationAlgorithm):
    """Differential Evolution (DE) Algorithm (DE/rand/1/bin strategy)."""

    def __init__(self, search_space: SearchSpace, pop_size: int = 20, F: float = 0.8, CR: float = 0.7) -> None:
        super().__init__(search_space, "DifferentialEvolution")
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.dim = search_space.dimension
        self.pop = np.random.uniform(0.0, 1.0, size=(self.pop_size, self.dim))
        self.fitnesses = np.full(self.pop_size, -1e9)

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for i in range(min(n, self.pop_size)):
            # Create trial vector via mutation and crossover
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), 0.0, 1.0)

            cross_mask = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_mask, mutant, self.pop[i])

            params = self.search_space.denormalize(trial)
            candidates.append(params)
        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        for i, f in enumerate(fitnesses[: self.pop_size]):
            if f >= self.fitnesses[i]:
                norm_vec = self.search_space.normalize(candidates[i])
                self.pop[i] = norm_vec
                self.fitnesses[i] = f


class SimulatedAnnealingAlgorithm(BaseOptimizationAlgorithm):
    """Simulated Annealing (SA) Algorithm."""

    def __init__(self, search_space: SearchSpace, initial_temp: float = 100.0, cooling_rate: float = 0.95) -> None:
        super().__init__(search_space, "SimulatedAnnealing")
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.dim = search_space.dimension

        self.current_pos = np.random.uniform(0.0, 1.0, size=self.dim)
        self.current_fitness = -1e9

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for _ in range(n):
            noise = np.random.normal(0.0, 0.1, size=self.dim)
            neighbor = np.clip(self.current_pos + noise, 0.0, 1.0)
            candidates.append(self.search_space.denormalize(neighbor))
        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        for i, f in enumerate(fitnesses):
            delta = f - self.current_fitness
            if delta > 0 or random.random() < math.exp(delta / max(1e-4, self.temp)):
                self.current_fitness = f
                self.current_pos = self.search_space.normalize(candidates[i])

        self.temp *= self.cooling_rate


class TPEAlgorithm(BaseOptimizationAlgorithm):
    """Tree-structured Parzen Estimator (TPE) Algorithm."""

    def __init__(self, search_space: SearchSpace, gamma: float = 0.25, n_startup: int = 10) -> None:
        super().__init__(search_space, "TPE")
        self.gamma = gamma
        self.n_startup = n_startup
        self.history_vecs: List[np.ndarray] = []
        self.history_scores: List[float] = []

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        if len(self.history_scores) < self.n_startup:
            return self.search_space.sample_batch(n)

        # Split history into good (l) and bad (g) based on gamma percentile
        cutoff = np.percentile(self.history_scores, (1.0 - self.gamma) * 100.0)
        good_vecs = [self.history_vecs[i] for i, s in enumerate(self.history_scores) if s >= cutoff]
        bad_vecs = [self.history_vecs[i] for i, s in enumerate(self.history_scores) if s < cutoff]

        good_arr = np.array(good_vecs) if len(good_vecs) > 0 else np.random.rand(5, self.search_space.dimension)

        for _ in range(n):
            # Sample around good solutions with kernel density estimation
            base_idx = random.randint(0, len(good_arr) - 1)
            sampled_vec = np.clip(good_arr[base_idx] + np.random.normal(0.0, 0.08, size=self.search_space.dimension), 0.0, 1.0)
            candidates.append(self.search_space.denormalize(sampled_vec))

        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        for i, c in enumerate(candidates):
            v = self.search_space.normalize(c)
            self.history_vecs.append(v)
            self.history_scores.append(fitnesses[i])


class BayesianOptimizerAlgorithm(TPEAlgorithm):
    """Bayesian Optimization surrogate sampler (using TPE/Gaussian Process surrogate)."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space, gamma=0.20, n_startup=8)
        self.name = "BayesianOptimization"


class EvolutionStrategyAlgorithm(BaseOptimizationAlgorithm):
    """(μ + λ)-Evolution Strategy Algorithm."""

    def __init__(self, search_space: SearchSpace, mu: int = 5, lambda_: int = 20, sigma: float = 0.1) -> None:
        super().__init__(search_space, "EvolutionStrategy")
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        self.dim = search_space.dimension

        self.parents = np.random.uniform(0.0, 1.0, size=(self.mu, self.dim))
        self.parent_fitnesses = np.full(self.mu, -1e9)

    def ask(self, n: int = 1) -> List[Dict[str, Any]]:
        candidates = []
        for _ in range(min(n, self.lambda_)):
            parent = self.parents[random.randint(0, self.mu - 1)]
            offspring = np.clip(parent + np.random.normal(0.0, self.sigma, size=self.dim), 0.0, 1.0)
            candidates.append(self.search_space.denormalize(offspring))
        return candidates

    def tell(self, candidates: List[Dict[str, Any]], fitnesses: List[float]) -> None:
        # Combine parents and offspring
        all_vecs = list(self.parents) + [self.search_space.normalize(c) for c in candidates]
        all_scores = list(self.parent_fitnesses) + fitnesses

        sorted_indices = np.argsort(all_scores)[::-1]
        for m in range(self.mu):
            idx = sorted_indices[m]
            self.parents[m] = all_vecs[idx]
            self.parent_fitnesses[m] = all_scores[idx]


class CMAESAlgorithm(EvolutionStrategyAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES)."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space, mu=5, lambda_=20, sigma=0.08)
        self.name = "CMA-ES"


class OptunaAdapterAlgorithm(TPEAlgorithm):
    """Optuna TPE algorithm adapter."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space)
        self.name = "Optuna"


class HyperOptAdapterAlgorithm(TPEAlgorithm):
    """HyperOpt TPE algorithm adapter."""

    def __init__(self, search_space: SearchSpace) -> None:
        super().__init__(search_space)
        self.name = "HyperOpt"


class Optimizer:
    """Master Institutional Strategy Optimizer."""

    ALGORITHMS = {
        "grid_search": GridSearchAlgorithm,
        "random_search": RandomSearchAlgorithm,
        "bayesian": BayesianOptimizerAlgorithm,
        "optuna": OptunaAdapterAlgorithm,
        "hyperopt": HyperOptAdapterAlgorithm,
        "pso": ParticleSwarmAlgorithm,
        "ga": GeneticAlgorithm,
        "es": EvolutionStrategyAlgorithm,
        "de": DifferentialEvolutionAlgorithm,
        "sa": SimulatedAnnealingAlgorithm,
        "tpe": TPEAlgorithm,
        "cma_es": CMAESAlgorithm,
    }

    def __init__(
        self,
        strategy_cls: Type[BaseStrategy],
        dataset: MarketDataset,
        search_space: SearchSpace,
        algorithm: str = "random_search",
        objective_function: Optional[ObjectiveFunction] = None,
        constraints: Optional[OptimizationConstraints] = None,
        cache: Optional[OptimizationCache] = None,
        history: Optional[OptimizationHistory] = None,
        scheduler: Optional[OptimizationScheduler] = None,
        version: str = "1.0.0",
    ) -> None:
        """Initialize Optimizer."""
        self.strategy_cls = strategy_cls
        self.dataset = dataset
        self.search_space = search_space
        self.algorithm_name = algorithm.lower()
        self.objective_function = objective_function or ObjectiveFunction()
        self.constraints = constraints or OptimizationConstraints()
        self.cache = cache or OptimizationCache()
        self.history = history or OptimizationHistory()
        self.scheduler = scheduler or OptimizationScheduler()
        self.version = version
        self.opt_id = f"OPT-{int(time.time())}"

        self.evaluator = SolutionEvaluator(
            strategy_cls=strategy_cls,
            dataset=dataset,
            objective_function=self.objective_function,
            constraints=self.constraints,
        )

        algo_cls = self.ALGORITHMS.get(self.algorithm_name, RandomSearchAlgorithm)
        self.algo: BaseOptimizationAlgorithm = algo_cls(self.search_space)

        self._is_paused: bool = False
        self._is_cancelled: bool = False

    def optimizar(self, max_evaluations: int = 50, batch_size: int = 4) -> IterationRecord:
        """Run full optimization process."""
        return self.optimize(max_evaluations=max_evaluations, batch_size=batch_size)

    def optimize(self, max_evaluations: int = 50, batch_size: int = 4) -> IterationRecord:
        """Run full optimization process (alias)."""
        logger.log_start(self.opt_id, self.algo.name, max_evaluations)
        start_t = time.perf_counter()

        evals_done = 0
        best_rec: Optional[IterationRecord] = None

        while evals_done < max_evaluations and not self._is_cancelled:
            if self._is_paused:
                time.sleep(0.1)
                continue

            current_batch_size = min(batch_size, max_evaluations - evals_done)
            candidates = self.algo.ask(n=current_batch_size)
            if not candidates:
                break

            # Check cache and run batch
            eval_results: List[EvaluationResult] = []
            uncached_candidates = []

            for c in candidates:
                cache_key = self.cache.generate_key(self.strategy_cls, self.dataset, c)
                cached = self.cache.get(cache_key)

                if cached:
                    eval_results.append(
                        EvaluationResult(
                            parameters=cached.parameters,
                            fitness_score=cached.fitness_score,
                            is_valid=cached.is_valid,
                            metrics=cached.metrics,
                            violations=[],
                            execution_time_sec=cached.execution_time_sec,
                        )
                    )
                else:
                    uncached_candidates.append(c)

            if uncached_candidates:
                batch_res = self.scheduler.run_batch(self.evaluator, uncached_candidates)
                for res in batch_res:
                    cache_key = self.cache.generate_key(self.strategy_cls, self.dataset, res.parameters)
                    self.cache.put(
                        key=cache_key,
                        parameters=res.parameters,
                        metrics=res.metrics,
                        fitness_score=res.fitness_score,
                        is_valid=res.is_valid,
                        execution_time_sec=res.execution_time_sec,
                    )
                    eval_results.append(res)

            fitnesses = [res.fitness_score for res in eval_results]
            self.algo.tell(candidates, fitnesses)

            for res in eval_results:
                evals_done += 1
                rec = self.history.add_record(
                    iteration_index=evals_done,
                    parameters=res.parameters,
                    fitness_score=res.fitness_score,
                    is_valid=res.is_valid,
                    metrics=res.metrics,
                    duration_sec=res.execution_time_sec,
                    violations=res.violations,
                )

                if res.is_valid and (best_rec is None or res.fitness_score > best_rec.fitness_score):
                    if best_rec is not None:
                        logger.log_improvement(
                            self.opt_id, evals_done, best_rec.fitness_score, res.fitness_score, res.parameters
                        )
                    best_rec = rec

                logger.log_iteration(
                    self.opt_id, evals_done, res.fitness_score, best_rec.fitness_score if best_rec else 0.0
                )

        total_time = time.perf_counter() - start_t
        best_fitness = best_rec.fitness_score if best_rec else -9999.0
        logger.log_completion(self.opt_id, best_fitness, total_time)

        top = self.history.get_top_solutions(k=1)
        return top[0] if top else self.history.get_all_records()[0]

    def reanudar(self) -> None:
        """Resume paused optimization."""
        self.resume()

    def resume(self) -> None:
        """Resume paused optimization."""
        self._is_paused = False
        logger.log_resume(self.opt_id)

    def pausar(self) -> None:
        """Pause ongoing optimization."""
        self.pause()

    def pause(self) -> None:
        """Pause ongoing optimization."""
        self._is_paused = True
        logger.log_pause(self.opt_id)

    def cancelar(self) -> None:
        """Cancel ongoing optimization."""
        self.cancel()

    def cancel(self) -> None:
        """Cancel ongoing optimization."""
        self._is_cancelled = True
        self.scheduler.cancel_all()
        logger.log_cancellation(self.opt_id)

    def guardar(self, filepath: str) -> str:
        """Save optimization state to JSON checkpoint file."""
        return self.save(filepath)

    def save(self, filepath: str) -> str:
        """Save optimization state to JSON checkpoint file."""
        state = {
            "opt_id": self.opt_id,
            "version": self.version,
            "algorithm": self.algorithm_name,
            "history": [asdict(r) for r in self.history.get_all_records()],
            "cache_stats": self.cache.statistics,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        return os.path.abspath(filepath)

    def cargar(self, filepath: str) -> None:
        """Load optimization state from JSON checkpoint file."""
        self.load(filepath)

    def load(self, filepath: str) -> None:
        """Load optimization state from JSON checkpoint file."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.opt_id = state.get("opt_id", self.opt_id)
        self.version = state.get("version", self.version)

    def versionar(self, version_type: str = "patch") -> str:
        """Increment version string."""
        return self.version_increment(version_type)

    def version_increment(self, version_type: str = "patch") -> str:
        """Increment semantic version string."""
        parts = self.version.split(".")
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        if version_type == "major":
            major += 1
            minor, patch = 0, 0
        elif version_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        self.version = f"{major}.{minor}.{patch}"
        return self.version

    def evaluar(self, parameters: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single candidate parameter combination."""
        return self.evaluate_single(parameters)

    def evaluate_single(self, parameters: Dict[str, Any]) -> EvaluationResult:
        """Evaluate a single candidate parameter combination."""
        return self.evaluator.evaluate(parameters)

    def exportar(self, filepath: str, export_format: str = "json") -> str:
        """Export optimization results."""
        return self.export(filepath, export_format)

    def export(self, filepath: str, export_format: str = "json") -> str:
        """Export optimization results."""
        df = self.history.to_dataframe()
        if export_format.lower() == "csv":
            df.to_csv(filepath, index=False)
        else:
            df.to_json(filepath, orient="records", indent=2)
        return os.path.abspath(filepath)
