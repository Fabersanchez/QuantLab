"""
QuantLab Cross-Asset Correlation Analyzer.

Calculates Pearson, Spearman rank, Kendall tau correlation matrices, and rolling correlation series
for multi-asset portfolios.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Institutional Cross-Asset Correlation Analyzer."""

    @staticmethod
    def compute_correlation_matrix(returns_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        """Compute asset returns correlation matrix (Pearson, Spearman, or Kendall).

        Args:
            returns_df: DataFrame of asset returns.
            method: Method identifier ('pearson', 'spearman', 'kendall').

        Returns:
            Correlation DataFrame matrix.
        """
        return returns_df.corr(method=method)

    @staticmethod
    def compute_rolling_correlation(
        series_a: pd.Series, series_b: pd.Series, window: int = 63
    ) -> pd.Series:
        """Compute rolling correlation series between two assets.

        Args:
            series_a: Asset A return series.
            series_b: Asset B return series.
            window: Rolling window size.

        Returns:
            Rolling correlation Series.
        """
        return series_a.rolling(window).corr(series_b)
