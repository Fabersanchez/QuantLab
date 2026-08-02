"""
QuantLab Feature Generator.

Provides automated feature generation algorithms across Price, Volume, Volatility,
Time, Microstructure, Order Flow, Statistical, Sentiment, and Cross-Asset categories.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class BaseFeatureGenerator(ABC):
    """Abstract Base Class for feature generator modules."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Return the category identifier for generated features."""
        pass

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute and attach generated features to DataFrame copy."""
        pass


class PriceFeatureGenerator(BaseFeatureGenerator):
    """Generates Price-derived predictive features."""

    @property
    def category(self) -> str:
        return "Price"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)
        cols = {c.lower(): c for c in df.columns}

        if "close" in cols:
            c = df[cols["close"]]
            features_df["log_return"] = np.log(c / c.shift(1))
            features_df["pct_change"] = c.pct_change()
            features_df["price_ratio_sma5"] = c / c.rolling(5).mean()

        if all(k in cols for k in ["high", "low", "open", "close"]):
            h = df[cols["high"]]
            l = df[cols["low"]]
            o = df[cols["open"]]
            c = df[cols["close"]]

            features_df["hl_spread"] = (h - l) / (c + 1e-8)
            features_df["co_spread"] = (c - o) / (c + 1e-8)

        return features_df


class VolumeFeatureGenerator(BaseFeatureGenerator):
    """Generates Volume-derived predictive features."""

    @property
    def category(self) -> str:
        return "Volume"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)
        cols = {c.lower(): c for c in df.columns}

        if "volume" in cols:
            v = df[cols["volume"]]
            features_df["volume_pct_change"] = v.pct_change()
            features_df["volume_sma5_ratio"] = v / (v.rolling(5).mean() + 1e-8)

            if "close" in cols:
                c = df[cols["close"]]
                ret = c.pct_change()
                features_df["volume_price_trend"] = (ret * v).cumsum()

        return features_df


class VolatilityFeatureGenerator(BaseFeatureGenerator):
    """Generates Volatility-derived predictive features."""

    @property
    def category(self) -> str:
        return "Volatility"

    def generate(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)
        cols = {c.lower(): c for c in df.columns}

        if "close" in cols:
            c = df[cols["close"]]
            ret = np.log(c / c.shift(1))
            features_df[f"volatility_rolling_{window}"] = ret.rolling(
                window
            ).std() * np.sqrt(252)

        if all(k in cols for k in ["high", "low"]):
            h = df[cols["high"]]
            l = df[cols["low"]]
            # Parkinson Volatility Proxy
            features_df[f"parkinson_vol_{window}"] = np.sqrt(
                (1.0 / (4.0 * window * np.log(2.0)))
                * (np.log(h / (l + 1e-8)) ** 2).rolling(window).sum()
            )

        return features_df


class TimeFeatureGenerator(BaseFeatureGenerator):
    """Generates Time/Calendar predictive features."""

    @property
    def category(self) -> str:
        return "Time"

    def generate(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)

        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col])
            features_df["day_of_week"] = ts.dt.dayofweek
            features_df["hour_of_day"] = ts.dt.hour
            features_df["minute_of_hour"] = ts.dt.minute
            features_df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
            features_df["sin_hour"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
            features_df["cos_hour"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
            features_df["day_of_week"] = ts.dayofweek
            features_df["hour_of_day"] = ts.hour
            features_df["minute_of_hour"] = ts.minute
            features_df["is_weekend"] = ts.dayofweek.isin([5, 6]).astype(int)

        return features_df


class StatisticalFeatureGenerator(BaseFeatureGenerator):
    """Generates Statistical time-series predictive features."""

    @property
    def category(self) -> str:
        return "Statistical"

    def generate(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        features_df = pd.DataFrame(index=df.index)
        cols = {c.lower(): c for c in df.columns}

        if "close" in cols:
            c = df[cols["close"]]
            mean = c.rolling(window).mean()
            std = c.rolling(window).std()
            features_df[f"zscore_{window}"] = (c - mean) / (std + 1e-8)
            features_df[f"skewness_{window}"] = c.rolling(window).skew()
            features_df[f"kurtosis_{window}"] = c.rolling(window).kurt()

        return features_df


class FeatureGenerator:
    """Master Feature Generator orchestrating all category generators."""

    def __init__(self) -> None:
        self._generators: List[BaseFeatureGenerator] = [
            PriceFeatureGenerator(),
            VolumeFeatureGenerator(),
            VolatilityFeatureGenerator(),
            TimeFeatureGenerator(),
            StatisticalFeatureGenerator(),
        ]

    def register_generator(self, generator: BaseFeatureGenerator) -> None:
        """Register a custom category feature generator."""
        self._generators.append(generator)

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all registered generators and merge features with original DataFrame."""
        result_df = df.copy()
        generated_frames = []

        for gen in self._generators:
            feat_df = gen.generate(df)
            if not feat_df.empty:
                generated_frames.append(feat_df)

        if generated_frames:
            all_feats = pd.concat(generated_frames, axis=1)
            result_df = pd.concat([result_df, all_feats], axis=1)

        return result_df
