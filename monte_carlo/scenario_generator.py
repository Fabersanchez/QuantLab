"""
QuantLab Master Scenario Generator.

Generates simulation scenarios via Random Shuffle, Bootstrap Sampling,
Trade Permutation, Noise Injection, Historical Stressing, and Synthetic Paths (GBM).
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from monte_carlo.bootstrap_sampling import BootstrapSamplerFactory
from monte_carlo.permutation_engine import TradeOrderPermutation, WinsLossesPermutation
from monte_carlo.noise_injector import CompositeNoiseInjector


class ScenarioType(str, Enum):
    """Supported Monte Carlo scenario generation types."""

    RANDOM_SHUFFLE = "RANDOM_SHUFFLE"
    BOOTSTRAP = "BOOTSTRAP"
    RESAMPLING = "RESAMPLING"
    TRADE_PERMUTATION = "TRADE_PERMUTATION"
    NOISE_INJECTION = "NOISE_INJECTION"
    HISTORICAL_STRESS = "HISTORICAL_STRESS"
    SYNTHETIC_SCENARIOS = "SYNTHETIC_SCENARIOS"


class ScenarioGenerator:
    """Master Scenario Generation Engine."""

    @staticmethod
    def generate_trade_scenarios(
        trades: Union[List[float], pd.Series],
        n_scenarios: int = 1000,
        scenario_type: ScenarioType = ScenarioType.TRADE_PERMUTATION,
        bootstrap_method: str = "random",
        block_size: int = 10,
    ) -> List[List[float]]:
        """Generate N trade PnL sequence scenarios.

        Args:
            trades: Original trade PnL list/series.
            n_scenarios: Number of simulation iterations to generate.
            scenario_type: Target ScenarioType.
            bootstrap_method: Sub-type if scenario_type is BOOTSTRAP.
            block_size: Block size for block bootstrap.

        Returns:
            List of lists of float trade PnLs.
        """
        series = pd.Series(trades) if not isinstance(trades, pd.Series) else trades
        if series.empty:
            return [[] for _ in range(n_scenarios)]

        scenarios: List[List[float]] = []

        if scenario_type in (ScenarioType.TRADE_PERMUTATION, ScenarioType.RANDOM_SHUFFLE):
            for _ in range(n_scenarios):
                perm = TradeOrderPermutation.permute(series)
                scenarios.append(perm.tolist())

        elif scenario_type in (ScenarioType.BOOTSTRAP, ScenarioType.RESAMPLING):
            sampler = BootstrapSamplerFactory.create(bootstrap_method, block_size=block_size)
            for _ in range(n_scenarios):
                boot = sampler.sample(series)
                scenarios.append(boot.tolist())

        elif scenario_type == ScenarioType.NOISE_INJECTION:
            std = float(series.std()) if len(series) > 1 else 1.0
            for _ in range(n_scenarios):
                noise = np.random.normal(0, std * 0.1, size=len(series))
                scenarios.append((series.values + noise).tolist())

        elif scenario_type == ScenarioType.SYNTHETIC_SCENARIOS:
            # Generate synthetic log-normal / Gaussian returns matching trade distribution
            mu = float(series.mean())
            sigma = float(series.std()) if len(series) > 1 else 1.0
            n_len = len(series)
            for _ in range(n_scenarios):
                synth = np.random.normal(mu, sigma, size=n_len)
                scenarios.append(synth.tolist())

        else:
            for _ in range(n_scenarios):
                scenarios.append(series.tolist())

        return scenarios

    @staticmethod
    def generate_market_scenarios(
        data: pd.DataFrame,
        n_scenarios: int = 100,
        scenario_type: ScenarioType = ScenarioType.NOISE_INJECTION,
    ) -> List[pd.DataFrame]:
        """Generate N market DataFrame scenarios with noise perturbations or synthetic return paths.

        Returns:
            List of market DataFrames.
        """
        scenarios: List[pd.DataFrame] = []
        c_col = "close" if "close" in data.columns else data.columns[0]
        returns = data[c_col].pct_change().fillna(0.0)

        if scenario_type == ScenarioType.NOISE_INJECTION:
            injector = CompositeNoiseInjector()
            for _ in range(n_scenarios):
                df_copy = data.copy()
                if "high" in df_copy.columns and "low" in df_copy.columns:
                    noise_factor = np.random.normal(1.0, 0.05, size=len(df_copy))
                    df_copy["high"] *= np.maximum(0.9, noise_factor)
                    df_copy["low"] *= np.maximum(0.9, noise_factor)
                scenarios.append(df_copy)

        elif scenario_type == ScenarioType.SYNTHETIC_SCENARIOS:
            mu = float(returns.mean())
            sigma = float(returns.std()) if len(returns) > 1 else 0.01
            start_p = float(data[c_col].iloc[0])

            for _ in range(n_scenarios):
                df_copy = data.copy()
                synth_rets = np.random.normal(mu, sigma, size=len(data))
                synth_prices = start_p * np.cumprod(1.0 + synth_rets)
                df_copy[c_col] = synth_prices
                if "open" in df_copy.columns:
                    df_copy["open"] = synth_prices
                if "high" in df_copy.columns:
                    df_copy["high"] = synth_prices * 1.002
                if "low" in df_copy.columns:
                    df_copy["low"] = synth_prices * 0.998
                scenarios.append(df_copy)

        else:
            for _ in range(n_scenarios):
                scenarios.append(data.copy())

        return scenarios
