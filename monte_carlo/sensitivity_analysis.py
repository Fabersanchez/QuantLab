"""
QuantLab Parameter Sensitivity Analyzer.

Evaluates performance sensitivity and elasticity against variations in Stop Loss,
Take Profit, Risk/Position Size, Spread, Commission, and Slippage levels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from backtesting.commission_model import FixedCommissionModel
from backtesting.slippage_model import FixedSlippageModel
from backtesting.spread_model import FixedSpreadModel
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy


@dataclass
class SensitivityPoint:
    """Dataclass representing a single parameter sensitivity evaluation point."""

    parameter_name: str
    parameter_value: float
    net_profit: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float


class SensitivityAnalyzer:
    """Institutional Parameter Sensitivity & Elasticity Analyzer."""

    @staticmethod
    def analyze_parameter_sensitivity(
        strategy_cls: Type[BaseStrategy],
        base_params: Dict[str, Any],
        dataset: MarketDataset,
        parameter_name: str,
        test_values: List[float],
        config: Optional[BacktestConfig] = None,
    ) -> List[SensitivityPoint]:
        """Evaluate sensitivity across variations of a single parameter.

        Args:
            strategy_cls: Target BaseStrategy class.
            base_params: Default strategy parameters dictionary.
            dataset: MarketDataset instance.
            parameter_name: Target parameter name to vary.
            test_values: List of numerical test values.
            config: Engine BacktestConfig options.

        Returns:
            List of SensitivityPoint objects.
        """
        points: List[SensitivityPoint] = []

        for val in test_values:
            params = base_params.copy()
            params[parameter_name] = val

            strat = strategy_cls(params=params)
            engine = BacktestEngine(config=config)
            engine.load_dataset(dataset)
            engine.load_strategy(strat)

            res = engine.start_simulation()

            pt = SensitivityPoint(
                parameter_name=parameter_name,
                parameter_value=float(val),
                net_profit=float(res.metrics.get("net_profit", 0.0)),
                sharpe_ratio=float(res.metrics.get("sharpe_ratio", 0.0)),
                max_drawdown_pct=float(res.metrics.get("max_drawdown_pct", 0.0)),
                win_rate=float(res.metrics.get("win_rate", 0.0)),
            )
            points.append(pt)

        return points

    @staticmethod
    def analyze_execution_cost_sensitivity(
        strategy: BaseStrategy,
        dataset: MarketDataset,
        spread_pips_range: List[float] = [0.0, 1.0, 2.0, 3.0, 5.0],
        slippage_pips_range: List[float] = [0.0, 0.5, 1.0, 2.0],
        commission_range: List[float] = [0.0, 1.0, 2.5, 5.0],
        config: Optional[BacktestConfig] = None,
    ) -> Dict[str, List[SensitivityPoint]]:
        """Evaluate sensitivity to spread, slippage, and commission costs.

        Returns:
            Dict mapping cost_type -> list of SensitivityPoint.
        """
        output: Dict[str, List[SensitivityPoint]] = {}

        # Spread sensitivity
        spread_pts = []
        for sp in spread_pips_range:
            engine = BacktestEngine(config=config)
            engine.load_dataset(dataset)
            engine.load_strategy(strategy)
            engine.set_spread_model(FixedSpreadModel(pips=sp))
            res = engine.start_simulation()
            spread_pts.append(
                SensitivityPoint(
                    parameter_name="spread_pips",
                    parameter_value=sp,
                    net_profit=float(res.metrics.get("net_profit", 0.0)),
                    sharpe_ratio=float(res.metrics.get("sharpe_ratio", 0.0)),
                    max_drawdown_pct=float(res.metrics.get("max_drawdown_pct", 0.0)),
                    win_rate=float(res.metrics.get("win_rate", 0.0)),
                )
            )
        output["spread"] = spread_pts

        # Slippage sensitivity
        slip_pts = []
        for sl in slippage_pips_range:
            engine = BacktestEngine(config=config)
            engine.load_dataset(dataset)
            engine.load_strategy(strategy)
            engine.set_slippage_model(FixedSlippageModel(pips=sl))
            res = engine.start_simulation()
            slip_pts.append(
                SensitivityPoint(
                    parameter_name="slippage_pips",
                    parameter_value=sl,
                    net_profit=float(res.metrics.get("net_profit", 0.0)),
                    sharpe_ratio=float(res.metrics.get("sharpe_ratio", 0.0)),
                    max_drawdown_pct=float(res.metrics.get("max_drawdown_pct", 0.0)),
                    win_rate=float(res.metrics.get("win_rate", 0.0)),
                )
            )
        output["slippage"] = slip_pts

        # Commission sensitivity
        comm_pts = []
        for cm in commission_range:
            engine = BacktestEngine(config=config)
            engine.load_dataset(dataset)
            engine.load_strategy(strategy)
            engine.set_commission_model(FixedCommissionModel(fee_per_order=cm))
            res = engine.start_simulation()
            comm_pts.append(
                SensitivityPoint(
                    parameter_name="commission_fee",
                    parameter_value=cm,
                    net_profit=float(res.metrics.get("net_profit", 0.0)),
                    sharpe_ratio=float(res.metrics.get("sharpe_ratio", 0.0)),
                    max_drawdown_pct=float(res.metrics.get("max_drawdown_pct", 0.0)),
                    win_rate=float(res.metrics.get("win_rate", 0.0)),
                )
            )
        output["commission"] = comm_pts

        return output

    @staticmethod
    def calculate_elasticity_score(points: List[SensitivityPoint]) -> float:
        """Calculate elasticity score (lower score = lower sensitivity / higher stability)."""
        if not points or len(points) < 2:
            return 0.0

        sharpes = [p.sharpe_ratio for p in points]
        if np.mean(sharpes) == 0:
            return 100.0

        cv = float(np.std(sharpes) / abs(np.mean(sharpes)))
        return float(min(100.0, cv * 100.0))
