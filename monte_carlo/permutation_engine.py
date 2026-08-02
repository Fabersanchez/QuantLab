"""
QuantLab Permutation Engine.

Provides random reshuffling / permutation algorithms for trade PnL sequences,
bar returns, win/loss distributions, and trade holding durations.
"""

import random
from typing import List, Optional, Union
import numpy as np
import pandas as pd


class TradeOrderPermutation:
    """Permutes the order of executed trade PnLs to evaluate sequence risk and drawdown dependency."""

    @staticmethod
    def permute(trades: Union[List[float], pd.Series]) -> pd.Series:
        """Randomly shuffle trade sequence.

        Args:
            trades: List or Series of trade PnL values.

        Returns:
            Shuffled pandas Series.
        """
        vals = np.array(trades, copy=True)
        np.random.shuffle(vals)
        return pd.Series(vals)


class ReturnsPermutation:
    """Permutes bar-by-bar percentage returns series."""

    @staticmethod
    def permute(returns: pd.Series) -> pd.Series:
        """Randomly shuffle bar returns while maintaining original index."""
        vals = returns.values.copy()
        np.random.shuffle(vals)
        return pd.Series(vals, index=returns.index)


class WinsLossesPermutation:
    """Permutes win and loss assignment while preserving the overall win rate and magnitude distributions."""

    @staticmethod
    def permute(trades: Union[List[float], pd.Series]) -> pd.Series:
        """Shuffle signs of trade PnLs."""
        vals = np.array(trades, copy=True)
        mags = np.abs(vals)
        signs = np.array([1 if x > 0 else -1 for x in vals])
        np.random.shuffle(signs)
        return pd.Series(mags * signs)


class SequencePermutation:
    """Permutes holding duration sequences or execution delay vectors."""

    @staticmethod
    def permute_sequences(durations: List[int]) -> List[int]:
        """Randomly permute duration array."""
        seq = list(durations)
        random.shuffle(seq)
        return seq
