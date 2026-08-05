"""
QuantLab Covariance Matrix Estimation & Shrinkage Engine.

Calculates sample covariance matrices, Ledoit-Wolf shrinkage covariance,
and Oracle Approximated Shrinkage (OAS) covariance matrices for portfolio optimization.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


class CovarianceAnalyzer:
    """Institutional Covariance Estimation & Shrinkage Engine."""

    @staticmethod
    def compute_sample_covariance(returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
        """Compute standard sample covariance matrix.

        Args:
            returns_df: DataFrame of asset returns.
            annualize: Whether to annualize covariance (x 252).

        Returns:
            Covariance DataFrame matrix.
        """
        cov = returns_df.cov()
        if annualize:
            cov *= 252.0
        return cov

    @staticmethod
    def compute_ledoit_wolf_covariance(returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
        """Compute Ledoit-Wolf shrinkage covariance matrix.

        Args:
            returns_df: DataFrame of asset returns.
            annualize: Whether to annualize covariance.

        Returns:
            Shrunk covariance DataFrame matrix.
        """
        clean_df = returns_df.dropna()
        lw = LedoitWolf()
        lw.fit(clean_df.values)
        cov_matrix = lw.covariance_
        if annualize:
            cov_matrix *= 252.0
        return pd.DataFrame(cov_matrix, index=clean_df.columns, columns=clean_df.columns)

    @staticmethod
    def compute_oas_covariance(returns_df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
        """Compute Oracle Approximated Shrinkage (OAS) covariance matrix."""
        clean_df = returns_df.dropna()
        oas = OAS()
        oas.fit(clean_df.values)
        cov_matrix = oas.covariance_
        if annualize:
            cov_matrix *= 252.0
        return pd.DataFrame(cov_matrix, index=clean_df.columns, columns=clean_df.columns)
