"""
QuantLab Strategy Pipeline.

Executes sequential strategy workflow:
Raw Data -> Validation -> Preparation -> Signal Generation -> Output Signal Matrix
"""

from typing import Tuple
import pandas as pd

from strategies.base_strategy import BaseStrategy
from strategies.strategy_validator import StrategyValidator, StrategyValidationReport


class StrategyPipeline:
    """Sequential Strategy Execution Pipeline."""

    def __init__(self) -> None:
        self.validator = StrategyValidator()

    def run(
        self, strategy: BaseStrategy, data: pd.DataFrame, timeframe: str = "1h"
    ) -> Tuple[pd.DataFrame, StrategyValidationReport]:
        """Run full strategy execution sequence.

        Args:
            strategy: BaseStrategy instance.
            data: Input market DataFrame.
            timeframe: Active timeframe identifier.

        Returns:
            Tuple of (DataFrame with attached signals, StrategyValidationReport).
        """
        # Step 1: Prepare data
        prepared_df = strategy.prepare(data)

        # Step 2: Validate
        report = self.validator.validate(strategy, prepared_df, timeframe=timeframe)
        if not report.is_valid:
            return pd.DataFrame(index=data.index), report

        # Step 3: Generate signals
        signal_df = strategy.generate_signal(prepared_df)

        output_df = pd.concat([prepared_df, signal_df], axis=1)
        return output_df, report
