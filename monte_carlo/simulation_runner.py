"""
QuantLab High-Throughput Monte Carlo Simulation Runner.

Executes thousands of Monte Carlo simulation runs (100, 500, 1,000, 5,000, 10,000+)
from trade PnL distributions or market scenarios.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from monte_carlo.scenario_generator import ScenarioGenerator, ScenarioType


@dataclass
class SimulationIterationResult:
    """Dataclass holding outputs for an individual Monte Carlo simulation iteration."""

    iteration_id: int
    equity_series: List[float]
    final_equity: float
    net_profit: float
    max_drawdown_amount: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    ruin_occurred: bool


class SimulationRunner:
    """Institutional High-Throughput Monte Carlo Simulation Engine."""

    def __init__(
        self, initial_capital: float = 100000.0, ruin_threshold_pct: float = 50.0
    ) -> None:
        """Initialize SimulationRunner.

        Args:
            initial_capital: Account starting balance.
            ruin_threshold_pct: Account drawdown percentage threshold defining ruin (e.g. 50.0%).
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        self.initial_capital = float(initial_capital)
        self.ruin_threshold_pct = float(ruin_threshold_pct)

    def run_simulations_from_trades(
        self,
        trades: Union[List[float], pd.Series],
        n_iterations: int = 1000,
        scenario_type: ScenarioType = ScenarioType.TRADE_PERMUTATION,
        bootstrap_method: str = "random",
        block_size: int = 10,
    ) -> List[SimulationIterationResult]:
        """Run Monte Carlo simulation across N iterations generated from trade PnLs.

        Args:
            trades: Series or list of historical trade PnL values.
            n_iterations: Number of Monte Carlo iterations (100, 500, 1000, 5000, 10000).
            scenario_type: Target scenario generation method.
            bootstrap_method: Sub-method for bootstrap.
            block_size: Block size for block sampling.

        Returns:
            List of SimulationIterationResult objects.
        """
        scenarios = ScenarioGenerator.generate_trade_scenarios(
            trades=trades,
            n_scenarios=n_iterations,
            scenario_type=scenario_type,
            bootstrap_method=bootstrap_method,
            block_size=block_size,
        )

        results: List[SimulationIterationResult] = []
        ruin_level = self.initial_capital * (1.0 - self.ruin_threshold_pct / 100.0)

        for idx, sc_trades in enumerate(scenarios):
            arr_trades = np.array(sc_trades, dtype=float)
            cum_pnl = np.cumsum(arr_trades) if len(arr_trades) > 0 else np.array([0.0])
            eq_series = (self.initial_capital + np.insert(cum_pnl, 0, 0.0)).tolist()

            final_eq = float(eq_series[-1])
            net_profit = final_eq - self.initial_capital

            # Drawdown calculation
            arr_eq = np.array(eq_series)
            running_max = np.maximum.accumulate(arr_eq)
            dd_amount = running_max - arr_eq
            max_dd_amount = float(np.max(dd_amount))
            max_dd_pct = float(np.max(dd_amount / np.where(running_max > 0, running_max, 1.0)) * 100.0)

            # Win rate & Ruin check
            total_trd = len(arr_trades)
            wins = np.sum(arr_trades > 0)
            win_rate = (float(wins) / total_trd * 100.0) if total_trd > 0 else 0.0

            ruin_occurred = bool(np.min(arr_eq) <= ruin_level)

            res = SimulationIterationResult(
                iteration_id=idx,
                equity_series=eq_series,
                final_equity=final_eq,
                net_profit=net_profit,
                max_drawdown_amount=max_dd_amount,
                max_drawdown_pct=max_dd_pct,
                total_trades=total_trd,
                win_rate=win_rate,
                ruin_occurred=ruin_occurred,
            )
            results.append(res)

        return results
