"""
QuantLab Extreme Market Stress Testing Engine.

Generates extreme stress scenarios: Market Crash, Flash Crash, High Volatility,
Overnight Price Gaps, Extreme Spread Expansion, and Low Liquidity Volume Collapse.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestConfig, BacktestEngine, BacktestResult
from data.market_dataset import MarketDataset
from strategies.base_strategy import BaseStrategy


class StressScenarioType(str, Enum):
    """Supported stress scenario types."""

    MARKET_CRASH = "MARKET_CRASH"
    FLASH_CRASH = "FLASH_CRASH"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    PRICE_GAP = "PRICE_GAP"
    EXTREME_SPREAD = "EXTREME_SPREAD"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


@dataclass
class StressTestResult:
    """Dataclass holding results of a single stress test scenario execution."""

    scenario_name: str
    scenario_type: StressScenarioType
    survived: bool
    net_profit: float
    max_drawdown_pct: float
    sharpe_ratio: float
    stop_out_occurred: bool
    backtest_result: BacktestResult


class StressTestRunner:
    """Institutional Stress Testing Engine."""

    @staticmethod
    def apply_stress_scenario(df: pd.DataFrame, scenario_type: StressScenarioType) -> pd.DataFrame:
        """Apply extreme stress modifications to market DataFrame.

        Args:
            df: Original market DataFrame.
            scenario_type: Target StressScenarioType.

        Returns:
            Stressed pandas DataFrame copy.
        """
        stressed = df.copy()
        n = len(stressed)
        if n < 10:
            return stressed

        c_col = "close" if "close" in stressed.columns else stressed.columns[0]
        o_col = "open" if "open" in stressed.columns else c_col
        h_col = "high" if "high" in stressed.columns else c_col
        l_col = "low" if "low" in stressed.columns else c_col
        v_col = "volume" if "volume" in stressed.columns else None

        mid_idx = n // 2

        if scenario_type == StressScenarioType.MARKET_CRASH:
            # 30% severe market drop starting from middle of dataset
            drop_factor = np.linspace(1.0, 0.70, n - mid_idx)
            for col in [o_col, h_col, l_col, c_col]:
                stressed.iloc[mid_idx:, stressed.columns.get_loc(col)] *= drop_factor

        elif scenario_type == StressScenarioType.FLASH_CRASH:
            # 15% sudden 3-bar flash crash and quick partial recovery
            fc_idx = mid_idx
            stressed.iloc[fc_idx, stressed.columns.get_loc(l_col)] *= 0.85
            stressed.iloc[fc_idx, stressed.columns.get_loc(c_col)] *= 0.87
            stressed.iloc[fc_idx + 1, stressed.columns.get_loc(l_col)] *= 0.86
            stressed.iloc[fc_idx + 1, stressed.columns.get_loc(c_col)] *= 0.90

        elif scenario_type == StressScenarioType.HIGH_VOLATILITY:
            # Expand bar high-low ranges by 3x
            range_half = (stressed[h_col] - stressed[l_col]) * 1.5
            stressed[h_col] = stressed[c_col] + range_half
            stressed[l_col] = np.maximum(1e-4, stressed[c_col] - range_half)

        elif scenario_type == StressScenarioType.PRICE_GAP:
            # 5% gap down open at mid point
            gap_idx = mid_idx
            stressed.iloc[gap_idx:, stressed.columns.get_loc(o_col)] *= 0.95
            stressed.iloc[gap_idx:, stressed.columns.get_loc(h_col)] *= 0.95
            stressed.iloc[gap_idx:, stressed.columns.get_loc(l_col)] *= 0.95
            stressed.iloc[gap_idx:, stressed.columns.get_loc(c_col)] *= 0.95

        elif scenario_type == StressScenarioType.EXTREME_SPREAD:
            # Expand spread column by 20x or add extreme spread
            if "spread" in stressed.columns:
                stressed["spread"] *= 20.0
            else:
                stressed["spread"] = (stressed[c_col] * 0.005)  # 0.5% spread

        elif scenario_type == StressScenarioType.LOW_LIQUIDITY:
            # 95% volume collapse
            if v_col and v_col in stressed.columns:
                stressed[v_col] *= 0.05

        return stressed

    @staticmethod
    def run_all_stress_tests(
        strategy: BaseStrategy,
        dataset: MarketDataset,
        config: Optional[BacktestConfig] = None,
    ) -> List[StressTestResult]:
        """Run strategy against all 6 extreme market stress scenarios.

        Returns:
            List of StressTestResult objects.
        """
        results: List[StressTestResult] = []
        base_df = dataset.data

        for st_type in StressScenarioType:
            stressed_df = StressTestRunner.apply_stress_scenario(base_df, st_type)
            stressed_dataset = MarketDataset(
                data=stressed_df,
                asset=dataset.metadata.asset,
                timeframe=dataset.metadata.timeframe,
                broker=dataset.metadata.broker,
            )

            engine = BacktestEngine(config=config)
            engine.load_dataset(stressed_dataset)
            engine.load_strategy(strategy)

            res = engine.start_simulation()

            net_p = float(res.metrics.get("net_profit", 0.0))
            max_dd = float(res.metrics.get("max_drawdown_pct", 0.0))
            sharpe = float(res.metrics.get("sharpe_ratio", 0.0))

            # Check if stop out or severe collapse occurred
            stop_out = any(t.exit_reason == "STOP_OUT_LIQUIDATION" for t in res.trade_log.get_all_trades())
            survived = not stop_out and max_dd < 50.0

            st_res = StressTestResult(
                scenario_name=st_type.value,
                scenario_type=st_type,
                survived=survived,
                net_profit=net_p,
                max_drawdown_pct=max_dd,
                sharpe_ratio=sharpe,
                stop_out_occurred=stop_out,
                backtest_result=res,
            )
            results.append(st_res)

        return results
