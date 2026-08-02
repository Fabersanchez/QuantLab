"""
QuantLab Institutional Walk Forward Optimization Engine Package.

Provides temporal validation window generators (Rolling, Expanding, Anchored, Sliding, Custom),
pluggable strategy optimization adapters (Grid Search, Random Search, Optuna, Bayesian, Genetic, PSO),
validation runners, robustness metric calculators (WFE, Stability, Overfitting), efficiency analyzers,
multi-format reporting (HTML, Markdown, PDF, JSON, CSV), visual analytics, and master WalkForwardEngine.
"""

from walk_forward.window_generator import (
    WindowSplit,
    BaseWindowGenerator,
    WindowGeneratorFactory,
)
from walk_forward.rolling_windows import (
    RollingWindowGenerator,
    SlidingWindowGenerator,
)
from walk_forward.expanding_windows import (
    ExpandingWindowGenerator,
)
from walk_forward.anchored_windows import (
    AnchoredWindowGenerator,
    CustomWindowGenerator,
)
from walk_forward.optimizer_interface import (
    BaseOptimizerAdapter,
    GridSearchOptimizerAdapter,
    RandomSearchOptimizerAdapter,
    OptunaOptimizerAdapter,
    BayesianOptimizerAdapter,
    GeneticOptimizerAdapter,
    ParticleSwarmOptimizerAdapter,
    OptimizerAdapterFactory,
)
from walk_forward.validation_runner import (
    ValidationStepResult,
    ValidationRunner,
)
from walk_forward.window_statistics import (
    WindowStatisticsCalculator,
)
from walk_forward.robustness_metrics import (
    RobustnessMetricsCalculator,
)
from walk_forward.efficiency import (
    EfficiencyAnalyzer,
)
from walk_forward.visualization import (
    WalkForwardVisualizer,
)
from walk_forward.report_generator import (
    WalkForwardReportGenerator,
)
from walk_forward.walkforward_engine import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
)

__all__ = [
    # Window Generators
    "WindowSplit",
    "BaseWindowGenerator",
    "WindowGeneratorFactory",
    "RollingWindowGenerator",
    "SlidingWindowGenerator",
    "ExpandingWindowGenerator",
    "AnchoredWindowGenerator",
    "CustomWindowGenerator",
    # Optimizer Interfaces
    "BaseOptimizerAdapter",
    "GridSearchOptimizerAdapter",
    "RandomSearchOptimizerAdapter",
    "OptunaOptimizerAdapter",
    "BayesianOptimizerAdapter",
    "GeneticOptimizerAdapter",
    "ParticleSwarmOptimizerAdapter",
    "OptimizerAdapterFactory",
    # Validation & Analytics
    "ValidationStepResult",
    "ValidationRunner",
    "WindowStatisticsCalculator",
    "RobustnessMetricsCalculator",
    "EfficiencyAnalyzer",
    "WalkForwardVisualizer",
    "WalkForwardReportGenerator",
    # Engine & Results
    "WalkForwardConfig",
    "WalkForwardEngine",
    "WalkForwardResult",
]
