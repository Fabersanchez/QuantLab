"""
QuantLab Portfolio Diversification & Concentration Analyzer.

Calculates Diversification Ratio, Effective Number of Assets (ENB),
Concentration Risk Index (Herfindahl-Hirschman Index HHI), and risk concentration.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DiversificationAnalyzer:
    """Institutional Portfolio Diversification Analyzer."""

    @staticmethod
    def compute_diversification_ratio(weights: Dict[str, float], cov_matrix: pd.DataFrame) -> float:
        """Compute Portfolio Diversification Ratio.

        Diversification Ratio = sum(w_i * sigma_i) / sqrt(w^T * Cov * w)

        Returns:
            Diversification ratio float (>= 1.0).
        """
        symbols = [s for s in weights.keys() if s in cov_matrix.columns]
        if not symbols:
            return 1.0

        w = np.array([weights[s] for s in symbols])
        if np.sum(w) <= 0:
            return 1.0
        w /= np.sum(w)

        cov = cov_matrix.loc[symbols, symbols].values
        vols = np.sqrt(np.diag(cov))

        weighted_vol = float(np.dot(w, vols))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

        if port_vol <= 0:
            return 1.0
        return float(weighted_vol / port_vol)

    @staticmethod
    def compute_effective_number_of_assets(weights: Dict[str, float]) -> float:
        """Compute Effective Number of Assets (ENB = 1 / sum(w_i^2)).

        Returns:
            Effective number of assets float (between 1.0 and N).
        """
        vals = np.array(list(weights.values()))
        if len(vals) == 0:
            return 0.0
        total = np.sum(vals)
        if total <= 0:
            return 0.0
        w = vals / total
        hhi = float(np.sum(w**2))
        return 1.0 / hhi if hhi > 0 else 0.0

    @staticmethod
    def compute_hhi(weights: Dict[str, float]) -> float:
        """Compute Herfindahl-Hirschman Index (HHI) concentration metric.

        Returns:
            HHI float (between 0.0 and 1.0).
        """
        vals = np.array(list(weights.values()))
        if len(vals) == 0:
            return 1.0
        total = np.sum(vals)
        if total <= 0:
            return 1.0
        w = vals / total
        return float(np.sum(w**2))
