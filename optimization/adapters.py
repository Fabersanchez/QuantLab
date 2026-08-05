"""
QuantLab Optimization Integration Adapters.

Provides clean, decoupled integration adapters transforming optimization outputs into
Research Engine experiments, Model Registry candidates, and Sentinel deployment specs.
"""

from typing import Any, Dict, Optional
from optimization.history import IterationRecord
from optimization.logger import get_optimization_logger
from research.experiment import Experiment, ExperimentStatus
from research.metadata import MetadataExtractor

logger = get_optimization_logger("IntegrationAdapters")


class OptimizationExperimentAdapter:
    """Decoupled adapter converting Optimization iteration records into Research Engine Experiments."""

    @staticmethod
    def to_experiment(
        record: IterationRecord,
        strategy_name: str = "OptimizedStrategy",
        asset_symbol: str = "EURUSD",
        timeframe: str = "1h",
        author: str = "OptimizationEngine",
    ) -> Experiment:
        """Convert IterationRecord into a fully formed Experiment object.

        Returns:
            Experiment instance.
        """
        metadata = MetadataExtractor.collect(
            broker="GenericBroker",
            random_seed=42,
        )

        exp = Experiment(
            name=f"OptCandidate_{strategy_name}_Eval{record.evaluation_id}",
            description=f"Optimal strategy parameter candidate from optimization eval {record.evaluation_id}.",
            author=author,
            asset=asset_symbol,
            timeframe=timeframe,
            parameters=record.parameters,
            status=ExperimentStatus.COMPLETED if record.is_valid else ExperimentStatus.REJECTED,
            execution_time=record.duration_sec,
            results={**record.metrics, "fitness_score": record.fitness_score},
            system_metadata=metadata.to_dict(),
        )

        logger.info(f"Adapted IterationRecord to Experiment UUID={exp.uuid}")
        return exp
