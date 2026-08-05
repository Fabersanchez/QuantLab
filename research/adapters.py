"""
QuantLab Research Engine Integration Adapters.

Provides clean, decoupled adapters transforming outputs from BacktestEngine, WalkForwardEngine,
MonteCarloEngine, MachineLearning, DeepLearning, and ReinforcementLearning into institutional
Experiment instances ready for registry storage, scoring, validation, and Sentinel promotion.
"""

from typing import Any, Dict, Optional
from research.experiment import Experiment, ExperimentStatus
from research.logger import get_research_logger
from research.metadata import MetadataExtractor

logger = get_research_logger("IntegrationAdapters")


class BacktestExperimentAdapter:
    """Decoupled adapter converting BacktestEngine outputs into institutional Experiments."""

    @staticmethod
    def to_experiment(
        backtest_result: Any,
        author: str = "QuantLabBacktester",
        broker: str = "GenericBroker",
        random_seed: int = 42,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Convert BacktestResult object into an Experiment instance.

        Args:
            backtest_result: BacktestResult object.
            author: Experiment author.
            broker: Broker identifier.
            random_seed: Random seed.
            dataset_info: Optional dataset metadata dict.

        Returns:
            Populated Experiment instance.
        """
        metrics = getattr(backtest_result, "metrics", {}) or {}
        statistics = getattr(backtest_result, "statistics", {}) or {}
        strategy_name = getattr(backtest_result, "strategy_name", "BacktestStrategy")
        asset_symbol = getattr(backtest_result, "asset_symbol", "EURUSD")
        timeframe = getattr(backtest_result, "timeframe", "1h")
        execution_time = float(getattr(backtest_result, "execution_time_seconds", 0.0))

        merged_results = {**metrics, **statistics}

        metadata = MetadataExtractor.collect(
            broker=broker,
            random_seed=random_seed,
        )

        exp = Experiment(
            name=f"Backtest_{strategy_name}_{asset_symbol}",
            description=f"Institutional backtest experiment for strategy {strategy_name}.",
            author=author,
            dataset=dataset_info or {"asset": asset_symbol, "timeframe": timeframe},
            broker=broker,
            asset=asset_symbol,
            timeframe=timeframe,
            parameters={"strategy_name": strategy_name},
            status=ExperimentStatus.COMPLETED,
            execution_time=execution_time,
            results=merged_results,
            random_seed=random_seed,
            system_metadata=metadata.to_dict(),
            resource_metrics={"backtest_time_sec": execution_time},
        )
        logger.info(f"Adapted BacktestResult to Experiment UUID={exp.uuid}")
        return exp


class WalkForwardExperimentAdapter:
    """Decoupled adapter converting WalkForwardEngine outputs into institutional Experiments."""

    @staticmethod
    def to_experiment(
        wf_result: Any,
        author: str = "QuantLabWalkForward",
        broker: str = "GenericBroker",
        random_seed: int = 42,
    ) -> Experiment:
        """Convert WalkForwardResult object into an Experiment instance.

        Args:
            wf_result: WalkForwardResult object.
            author: Experiment author.
            broker: Broker identifier.
            random_seed: Random seed.

        Returns:
            Populated Experiment instance.
        """
        strategy_name = getattr(wf_result, "strategy_name", "WFStrategy")
        asset_symbol = getattr(wf_result, "asset_symbol", "EURUSD")
        robustness = getattr(wf_result, "robustness_metrics", {}) or {}
        efficiency = getattr(wf_result, "efficiency_metrics", {}) or {}
        stats = getattr(wf_result, "statistics_summary", {}) or {}
        execution_time = float(getattr(wf_result, "execution_time_seconds", 0.0))

        merged_results = {**robustness, **efficiency, **stats}

        exp = Experiment(
            name=f"WalkForward_{strategy_name}_{asset_symbol}",
            description=f"Institutional Walk Forward analysis experiment for strategy {strategy_name}.",
            author=author,
            broker=broker,
            asset=asset_symbol,
            timeframe="1h",
            status=ExperimentStatus.COMPLETED,
            execution_time=execution_time,
            results=merged_results,
            random_seed=random_seed,
            resource_metrics={"walk_forward_time_sec": execution_time},
        )
        logger.info(f"Adapted WalkForwardResult to Experiment UUID={exp.uuid}")
        return exp


class MonteCarloExperimentAdapter:
    """Decoupled adapter converting MonteCarloEngine outputs into institutional Experiments."""

    @staticmethod
    def to_experiment(
        mc_result: Any,
        author: str = "QuantLabMonteCarlo",
        broker: str = "GenericBroker",
        random_seed: int = 42,
    ) -> Experiment:
        """Convert MonteCarloResult object into an Experiment instance.

        Args:
            mc_result: MonteCarloResult object.
            author: Experiment author.
            broker: Broker identifier.
            random_seed: Random seed.

        Returns:
            Populated Experiment instance.
        """
        strategy_name = getattr(mc_result, "strategy_name", "MCStrategy")
        asset_symbol = getattr(mc_result, "asset_symbol", "EURUSD")
        dist = getattr(mc_result, "distribution_metrics", {}) or {}
        prob = getattr(mc_result, "probability_metrics", {}) or {}
        rob = getattr(mc_result, "robustness_score", {}) or {}
        execution_time = float(getattr(mc_result, "execution_time_seconds", 0.0))

        merged_results = {**dist, **prob, **rob}

        exp = Experiment(
            name=f"MonteCarlo_{strategy_name}_{asset_symbol}",
            description=f"Institutional Monte Carlo robustness experiment for strategy {strategy_name}.",
            author=author,
            broker=broker,
            asset=asset_symbol,
            timeframe="1h",
            status=ExperimentStatus.COMPLETED,
            execution_time=execution_time,
            results=merged_results,
            random_seed=random_seed,
            resource_metrics={"monte_carlo_time_sec": execution_time},
        )
        logger.info(f"Adapted MonteCarloResult to Experiment UUID={exp.uuid}")
        return exp


class MLExperimentAdapter:
    """Decoupled adapter converting Machine Learning / Deep Learning / Reinforcement Learning runs into Experiments."""

    @staticmethod
    def to_experiment(
        ml_run: Any,
        author: str = "QuantLabML",
        broker: str = "GenericBroker",
        random_seed: int = 42,
    ) -> Experiment:
        """Convert ML ExperimentRun object into an Experiment instance.

        Args:
            ml_run: ExperimentRun object.
            author: Author identifier.
            broker: Broker identifier.
            random_seed: Random seed.

        Returns:
            Populated Experiment instance.
        """
        exp_name = getattr(ml_run, "experiment_name", "ML_Model_Experiment")
        model_type = getattr(ml_run, "model_type", "StandardML")
        hyperparams = getattr(ml_run, "hyperparameters", {}) or {}
        metrics = getattr(ml_run, "metrics", {}) or {}
        duration = float(getattr(ml_run, "duration_seconds", 0.0))

        exp = Experiment(
            name=f"ML_{exp_name}_{model_type}",
            description=f"Machine Learning experiment run for model {model_type}.",
            author=author,
            broker=broker,
            parameters=hyperparams,
            status=ExperimentStatus.COMPLETED,
            execution_time=duration,
            results=metrics,
            random_seed=random_seed,
        )
        logger.info(f"Adapted ML ExperimentRun to Experiment UUID={exp.uuid}")
        return exp
