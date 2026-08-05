"""
QuantLab Master Optimization Engine Package.

Provides institutional automated strategy parameter optimization using 12 classical,
heuristic, and AI-driven algorithms integrated with Backtesting, Walk Forward,
Monte Carlo, and Research Engine.
"""

from optimization.adapters import OptimizationExperimentAdapter
from optimization.cache import CacheEntry, OptimizationCache
from optimization.constraints import ConstraintRule, OptimizationConstraints
from optimization.evaluator import EvaluationResult, SolutionEvaluator
from optimization.exporter import OptimizationExporter
from optimization.history import IterationRecord, OptimizationHistory
from optimization.logger import OptimizationLogger, get_optimization_logger
from optimization.objective_function import ObjectiveFunction, ObjectiveWeight
from optimization.optimization_manager import OptimizationJob, OptimizationManager
from optimization.optimizer import (
    BayesianOptimizerAlgorithm,
    CMAESAlgorithm,
    DifferentialEvolutionAlgorithm,
    EvolutionStrategyAlgorithm,
    GeneticAlgorithm,
    GridSearchAlgorithm,
    HyperOptAdapterAlgorithm,
    Optimizer,
    OptunaAdapterAlgorithm,
    ParticleSwarmAlgorithm,
    RandomSearchAlgorithm,
    SimulatedAnnealingAlgorithm,
    TPEAlgorithm,
)
from optimization.parameter_space import (
    BooleanParameter,
    CategoricalParameter,
    ContinuousParameter,
    CustomParameter,
    DiscreteParameter,
    FloatParameter,
    IntegerParameter,
    LogScaleParameter,
    NormalParameter,
    Parameter,
    UniformParameter,
)
from optimization.reports import OptimizationReport, OptimizationReportEngine
from optimization.scheduler import OptimizationScheduler, OptimizationTask
from optimization.search_space import SearchSpace
from optimization.visualization import OptimizationVisualizer

__all__ = [
    "Optimizer",
    "OptimizationManager",
    "OptimizationJob",
    "SearchSpace",
    "Parameter",
    "IntegerParameter",
    "FloatParameter",
    "BooleanParameter",
    "CategoricalParameter",
    "DiscreteParameter",
    "ContinuousParameter",
    "LogScaleParameter",
    "UniformParameter",
    "NormalParameter",
    "CustomParameter",
    "OptimizationConstraints",
    "ConstraintRule",
    "ObjectiveFunction",
    "ObjectiveWeight",
    "SolutionEvaluator",
    "EvaluationResult",
    "OptimizationCache",
    "CacheEntry",
    "OptimizationHistory",
    "IterationRecord",
    "OptimizationScheduler",
    "OptimizationTask",
    "OptimizationExporter",
    "OptimizationReportEngine",
    "OptimizationReport",
    "OptimizationVisualizer",
    "OptimizationLogger",
    "get_optimization_logger",
    "OptimizationExperimentAdapter",
    "GridSearchAlgorithm",
    "RandomSearchAlgorithm",
    "BayesianOptimizerAlgorithm",
    "OptunaAdapterAlgorithm",
    "HyperOptAdapterAlgorithm",
    "ParticleSwarmAlgorithm",
    "GeneticAlgorithm",
    "EvolutionStrategyAlgorithm",
    "DifferentialEvolutionAlgorithm",
    "SimulatedAnnealingAlgorithm",
    "TPEAlgorithm",
    "CMAESAlgorithm",
]
