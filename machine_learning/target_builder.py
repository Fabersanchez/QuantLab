"""
QuantLab Target Variable Construction Engine.

Builds quantitative target labels for machine learning models:
Directional Classification (+1, 0, -1), Binary Classification (1, 0),
Continuous Return Regression, and Triple Barrier Labeling (De Prado framework).
"""

from typing import Optional
import numpy as np
import pandas as pd


class TargetBuilder:
    """Quantitative Target Variable Generator."""

    @staticmethod
    def build_binary_classification_target(
        df: pd.DataFrame, horizon: int = 1, price_col: str = "close"
    ) -> pd.Series:
        """Build binary target (1 if price increases horizon bars ahead, 0 otherwise).

        Args:
            df: Market DataFrame.
            horizon: Forward prediction horizon in bars.
            price_col: Price column name.

        Returns:
            pandas Series of binary 1 / 0 labels.
        """
        c = df[price_col]
        fwd_ret = (c.shift(-horizon) - c) / c
        target = np.where(fwd_ret > 0, 1, 0)
        return pd.Series(target, index=df.index, name="target_binary").iloc[:-horizon]

    @staticmethod
    def build_directional_target(
        df: pd.DataFrame,
        horizon: int = 1,
        threshold_pct: float = 0.001,
        price_col: str = "close",
    ) -> pd.Series:
        """Build 3-class directional target (+1 Long, 0 Neutral, -1 Short).

        Args:
            df: Market DataFrame.
            horizon: Forward prediction horizon in bars.
            threshold_pct: Decimal return threshold (e.g. 0.001 for 0.1%).
            price_col: Price column.

        Returns:
            pandas Series of +1, 0, -1 labels.
        """
        c = df[price_col]
        fwd_ret = (c.shift(-horizon) - c) / c
        target = np.where(
            fwd_ret > threshold_pct,
            1,
            np.where(fwd_ret < -threshold_pct, -1, 0),
        )
        return pd.Series(target, index=df.index, name="target_directional").iloc[:-horizon]

    @staticmethod
    def build_continuous_return_target(
        df: pd.DataFrame, horizon: int = 1, price_col: str = "close"
    ) -> pd.Series:
        """Build continuous regression target (n-bar forward percentage return)."""
        c = df[price_col]
        fwd_ret = (c.shift(-horizon) - c) / c
        return pd.Series(fwd_ret, index=df.index, name="target_return").iloc[:-horizon]

    @staticmethod
    def build_triple_barrier_target(
        df: pd.DataFrame,
        pt_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        vertical_barrier_bars: int = 10,
        price_col: str = "close",
    ) -> pd.Series:
        """Build Triple Barrier Labels (De Prado Framework).

        Labels:
            +1: Touches Profit-Taking (PT) barrier first.
            -1: Touches Stop-Loss (SL) barrier first.
             0: Expires at vertical barrier without hitting PT or SL.

        Args:
            df: Market DataFrame with 'high', 'low', 'close'.
            pt_multiplier: Take Profit ATR/std multiplier.
            sl_multiplier: Stop Loss ATR/std multiplier.
            vertical_barrier_bars: Time expiration barrier.
            price_col: Price column.

        Returns:
            pandas Series of +1, -1, 0 labels.
        """
        close = df[price_col]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close
        n = len(df)

        volatility = close.pct_change().rolling(20).std().fillna(0.01) * close

        labels = np.zeros(n, dtype=int)
        for i in range(n - vertical_barrier_bars):
            p0 = close.iloc[i]
            vol = volatility.iloc[i]
            if vol <= 0:
                vol = p0 * 0.001

            pt_barrier = p0 + (pt_multiplier * vol)
            sl_barrier = p0 - (sl_multiplier * vol)

            # Look forward within vertical barrier window
            sub_high = high.iloc[i + 1 : i + 1 + vertical_barrier_bars]
            sub_low = low.iloc[i + 1 : i + 1 + vertical_barrier_bars]

            label = 0
            for h_val, l_val in zip(sub_high, sub_low):
                if h_val >= pt_barrier:
                    label = 1
                    break
                if l_val <= sl_barrier:
                    label = -1
                    break

            labels[i] = label

        return pd.Series(labels, index=df.index, name="target_triple_barrier").iloc[:-vertical_barrier_bars]
