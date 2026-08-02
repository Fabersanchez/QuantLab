"""
QuantLab Institutional Monte Carlo & Robustness Engine Package.

Provides statistical bootstrap samplers (Standard, Non-overlapping Block, Moving Block, Stationary),
sequence permutation algorithms, execution noise injectors (Spread, Slippage, Commission, Latency, Volatility),
extreme market stress testing (Crash, Flash Crash, High Volatility, Gap, Extreme Spread, Low Liquidity),
parameter sensitivity analysis, scenario generators, high-throughput simulation runners (100 to 10,000+ iterations),
statistical distribution calculators, probabilistic risk metrics (PoP, PoR), confidence intervals (90%, 95%, 99%),
composite Institutional Robustness Score engine, visual analytics (Equity Fan Charts, Drawdown Fan Charts),
multi-format reporting (HTML, Markdown, PDF, JSON, CSV), and master MonteCarloEngine.
"""

from monte_carlo.bootstrap_sampling import (
    BaseBootstrapSampler,
    RandomReplacementBootstrap,
    BlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
    BootstrapSamplerFactory,
)
from monte_carlo.permutation_engine import (
    TradeOrderPermutation,
    ReturnsPermutation,
    WinsLossesPermutation,
    SequencePermutation,
)
from monte_carlo.noise_injector import (
    BaseNoiseInjector,
    SpreadNoiseInjector,
    SlippageNoiseInjector,
    CommissionNoiseInjector,
    LatencyNoiseInjector,
    VolatilityNoiseInjector,
    CompositeNoiseInjector,
)
from monte_carlo.stress_testing import (
    StressScenarioType,
    StressTestResult,
    StressTestRunner,
)
from monte_carlo.sensitivity_analysis import (
    SensitivityPoint,
    SensitivityAnalyzer,
)
from monte_carlo.scenario_generator import (
    ScenarioType,
    ScenarioGenerator,
)
from monte_carlo.simulation_runner import (
    SimulationIterationResult,
    SimulationRunner,
)
from monte_carlo.distribution_metrics import (
    DistributionMetricsCalculator,
)
from monte_carlo.probability_analysis import (
    ProbabilityAnalyzer,
)
from monte_carlo.confidence_intervals import (
    ConfidenceIntervalCalculator,
)
from monte_carlo.robustness_score import (
    InstitutionalRobustnessScore,
)
from monte_carlo.visualization import (
    MonteCarloVisualizer,
)
from monte_carlo.report_generator import (
    MonteCarloReportGenerator,
)
from monte_carlo.montecarlo_engine import (
    MonteCarloConfig,
    MonteCarloEngine,
    MonteCarloResult,
)

__all__ = [
    # Bootstrap & Permutation
    "BaseBootstrapSampler",
    "RandomReplacementBootstrap",
    "BlockBootstrap",
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "BootstrapSamplerFactory",
    "TradeOrderPermutation",
    "ReturnsPermutation",
    "WinsLossesPermutation",
    "SequencePermutation",
    # Noise Injectors & Stress Testing
    "BaseNoiseInjector",
    "SpreadNoiseInjector",
    "SlippageNoiseInjector",
    "CommissionNoiseInjector",
    "LatencyNoiseInjector",
    "VolatilityNoiseInjector",
    "CompositeNoiseInjector",
    "StressScenarioType",
    "StressTestResult",
    "StressTestRunner",
    "SensitivityPoint",
    "SensitivityAnalyzer",
    # Scenario & Simulation Runner
    "ScenarioType",
    "ScenarioGenerator",
    "SimulationIterationResult",
    "SimulationRunner",
    # Analytics & Probabilities
    "DistributionMetricsCalculator",
    "ProbabilityAnalyzer",
    "ConfidenceIntervalCalculator",
    "InstitutionalRobustnessScore",
    # Visuals & Reporting
    "MonteCarloVisualizer",
    "MonteCarloReportGenerator",
    # Engine & Results
    "MonteCarloConfig",
    "MonteCarloEngine",
    "MonteCarloResult",
]
