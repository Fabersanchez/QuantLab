"""
QuantLab Quantitative Machine Learning Data Labeler Engine.

Implements Triple Barrier Method (Take Profit, Stop Loss, Expiration Horizon),
Fixed Horizon Directional Classification (1 = Buy, -1 = Sell, 0 = Hold), and Continuous Return Regression labeling.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DataLabeler:
    """Institutional Machine Learning Data Labeling Engine."""

    @staticmethod
    def label_fixed_horizon_directional(
        df: pd.DataFrame, horizon: int = 5, threshold: float = 0.001
    ) -> pd.Series:
        """Create Fixed Horizon Directional Classification Labels (1 = Buy, -1 = Sell, 0 = Hold).

        Args:
            df: DataFrame containing 'close' column.
            horizon: Future bar horizon int.
            threshold: Return threshold float.

        Returns:
            Label Series (1, -1, 0).
        """
        if "close" not in df.columns or df.empty:
            return pd.Series(dtype=int)

        closes = df["close"].values
        future_return = (np.roll(closes, -horizon) - closes) / closes
        # Set last horizon elements to 0
        future_return[-horizon:] = 0.0

        labels = np.zeros(len(closes), dtype=int)
        labels[future_return >= threshold] = 1
        labels[future_return <= -threshold] = -1

        return pd.Series(labels, index=df.index, name="label_directional")

    @staticmethod
    def label_triple_barrier(
        df: pd.DataFrame,
        pt_ratio: float = 0.01,
        sl_ratio: float = 0.01,
        max_holding: int = 10,
    ) -> pd.Series:
        """Create Triple Barrier Method ML labels (1 = Take Profit hit, -1 = Stop Loss hit, 0 = Timeout).

        Args:
            df: DataFrame containing 'high', 'low', 'close'.
            pt_ratio: Profit Target ratio float.
            sl_ratio: Stop Loss ratio float.
            max_holding: Maximum holding horizon bars.

        Returns:
            Label Series (1, -1, 0).
        """
        if df.empty or "close" not in df.columns:
            return pd.Series(dtype=int)

        closes = df["close"].values
        highs = df.get("high", df["close"]).values
        lows = df.get("low", df["close"]).values
        n = len(closes)

        labels = np.zeros(n, dtype=int)

        for i in range(n - 1):
            entry_p = closes[i]
            pt_p = entry_p * (1.0 + pt_ratio)
            sl_p = entry_p * (1.0 - sl_ratio)

            end_j = min(i + max_holding + 1, n)
            label = 0

            for j in range(i + 1, end_j):
                if highs[j] >= pt_p:
                    label = 1
                    break
                elif lows[j] <= sl_p:
                    label = -1
                    break

            labels[i] = label

        return pd.Series(labels, index=df.index, name="label_triple_barrier")
