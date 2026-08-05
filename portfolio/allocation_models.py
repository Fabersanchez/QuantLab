"""
QuantLab Capital Allocation Models.

Provides 10 institutional capital allocation models:
1. Equal Weight (1/N)
2. Risk Parity
3. Minimum Variance
4. Maximum Diversification
5. Kelly Allocation
6. Hierarchical Risk Parity (HRP)
7. Black-Litterman
8. Mean-Variance Optimization (Markowitz MVO)
9. Equal Risk Contribution (ERC)
10. Custom Allocation
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list


class BaseAllocationModel(ABC):
    """Abstract Base Class for all Capital Allocation Models."""

    def __init__(self, name: str) -> None:
        """Initialize allocation model.

        Args:
            name: Allocation model identifier.
        """
        self.name = name

    @abstractmethod
    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        r"""Compute target capital allocation weights ($w_i \ge 0$, $\sum w_i = 1.0$).

        Returns:
            Dictionary mapping asset symbol to allocation weight float.
        """
        pass


class EqualWeightModel(BaseAllocationModel):
    """Equal Weight (1/N) Capital Allocation Model."""

    def __init__(self) -> None:
        super().__init__("EqualWeight")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}
        w = 1.0 / n
        return {sym: w for sym in asset_symbols}


class RiskParityModel(BaseAllocationModel):
    """Inverse Volatility / Risk Parity Capital Allocation Model."""

    def __init__(self) -> None:
        super().__init__("RiskParity")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if returns_df is not None and not returns_df.empty:
            vols = returns_df[asset_symbols].std().values
        elif cov_matrix is not None and not cov_matrix.empty:
            vols = np.sqrt(np.diag(cov_matrix.loc[asset_symbols, asset_symbols].values))
        else:
            vols = np.ones(n)

        vols = np.where(vols <= 0, 1e-4, vols)
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        return {asset_symbols[i]: float(weights[i]) for i in range(n)}


class MinimumVarianceModel(BaseAllocationModel):
    """Minimum Variance Optimization Capital Allocation Model (Minimizing w^T Sigma w)."""

    def __init__(self) -> None:
        super().__init__("MinimumVariance")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            cov = returns_df[asset_symbols].cov().values
        else:
            cov = np.eye(n)

        # Optimization
        def port_var(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(cov, w)))

        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        w0 = np.full(n, 1.0 / n)

        res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else w0
        weights = np.maximum(0.0, weights)
        weights /= np.sum(weights)
        return {asset_symbols[i]: float(weights[i]) for i in range(n)}


class MaximumDiversificationModel(BaseAllocationModel):
    """Maximum Diversification Ratio Capital Allocation Model."""

    def __init__(self) -> None:
        super().__init__("MaximumDiversification")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            cov = returns_df[asset_symbols].cov().values
        else:
            cov = np.eye(n)

        vols = np.sqrt(np.diag(cov))

        # Maximize (w^T vols) / sqrt(w^T cov w) -> Minimize negative
        def neg_div_ratio(w: np.ndarray) -> float:
            weighted_vol = float(np.dot(w, vols))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            if port_vol == 0:
                return 0.0
            return -(weighted_vol / port_vol)

        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        w0 = np.full(n, 1.0 / n)

        res = minimize(neg_div_ratio, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else w0
        weights = np.maximum(0.0, weights)
        weights /= np.sum(weights)
        return {asset_symbols[i]: float(weights[i]) for i in range(n)}


class KellyAllocationModel(BaseAllocationModel):
    """Kelly Criterion Fractional Position Sizing Capital Allocation Model."""

    def __init__(self, kelly_fraction: float = 0.5) -> None:
        super().__init__("KellyAllocation")
        self.kelly_fraction = kelly_fraction

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if expected_returns is not None and cov_matrix is not None:
            mu = expected_returns.loc[asset_symbols].values
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
            try:
                raw_kelly = np.dot(np.linalg.inv(cov), mu) * self.kelly_fraction
            except Exception:
                raw_kelly = np.full(n, 1.0 / n)
        else:
            raw_kelly = np.full(n, 1.0 / n)

        weights = np.maximum(0.0, raw_kelly)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.full(n, 1.0 / n)

        return {asset_symbols[i]: float(weights[i]) for i in range(n)}


class HRPAllocationModel(BaseAllocationModel):
    """Hierarchical Risk Parity (HRP) Capital Allocation Model."""

    def __init__(self) -> None:
        super().__init__("HierarchicalRiskParity")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            cov = returns_df[asset_symbols].cov().values
        else:
            cov = np.eye(n)

        # Correlation matrix
        vols = np.sqrt(np.diag(cov))
        vols_outer = np.outer(vols, vols)
        vols_outer[vols_outer == 0] = 1e-8
        corr = cov / vols_outer

        # Distance matrix
        dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        link = linkage(dist, method="single")
        sort_idx = leaves_list(link)

        # Inverse variance recursive bisection
        weights = pd.Series(1.0, index=range(n))
        items = [sort_idx]

        while len(items) > 0:
            items = [
                i[j:k]
                for i in items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(items), 2):
                c_items0 = items[i]
                c_items1 = items[i + 1]

                cov0 = cov[np.ix_(c_items0, c_items0)]
                cov1 = cov[np.ix_(c_items1, c_items1)]

                var0 = float(np.dot(np.full(len(c_items0), 1.0 / len(c_items0)), np.dot(cov0, np.full(len(c_items0), 1.0 / len(c_items0)))))
                var1 = float(np.dot(np.full(len(c_items1), 1.0 / len(c_items1)), np.dot(cov1, np.full(len(c_items1), 1.0 / len(c_items1)))))

                alpha = 1.0 - var0 / (var0 + var1) if (var0 + var1) > 0 else 0.5
                weights.iloc[c_items0] *= alpha
                weights.iloc[c_items1] *= 1.0 - alpha

        weights_arr = weights.values / np.sum(weights.values)
        return {asset_symbols[i]: float(weights_arr[i]) for i in range(n)}


class BlackLittermanModel(BaseAllocationModel):
    """Black-Litterman Model combining market equilibrium with investor views."""

    def __init__(self, tau: float = 0.05) -> None:
        super().__init__("BlackLitterman")
        self.tau = tau

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            cov = returns_df[asset_symbols].cov().values
        else:
            cov = np.eye(n)

        # Equilibrium returns (pi)
        w_eq = np.full(n, 1.0 / n)
        gamma = 2.5
        pi = gamma * np.dot(cov, w_eq)

        # Incorporate views if provided
        if views_dict:
            P = []
            q = []
            for idx, sym in enumerate(asset_symbols):
                if sym in views_dict:
                    row = np.zeros(n)
                    row[idx] = 1.0
                    P.append(row)
                    q.append(views_dict[sym])
            if P:
                P = np.array(P)
                q = np.array(q)
                omega = np.diag(np.diag(np.dot(P, np.dot(self.tau * cov, P.T))))
                omega[omega == 0] = 1e-4

                # BL expected returns formula
                term1 = np.linalg.inv(np.linalg.inv(self.tau * cov) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
                term2 = np.dot(np.linalg.inv(self.tau * cov), pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), q))
                mu_bl = np.dot(term1, term2)
            else:
                mu_bl = pi
        else:
            mu_bl = pi

        # Optimize MVO weights
        try:
            w_bl = np.dot(np.linalg.inv(cov), mu_bl)
            w_bl = np.maximum(0.0, w_bl)
            if np.sum(w_bl) > 0:
                w_bl /= np.sum(w_bl)
            else:
                w_bl = w_eq
        except Exception:
            w_bl = w_eq

        return {asset_symbols[i]: float(w_bl[i]) for i in range(n)}


class MeanVarianceModel(BaseAllocationModel):
    """Markowitz Mean-Variance Optimization Model (MVO)."""

    def __init__(self, risk_aversion: float = 1.0) -> None:
        super().__init__("MeanVariance")
        self.risk_aversion = risk_aversion

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        n = len(asset_symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and not cov_matrix.empty:
            cov = cov_matrix.loc[asset_symbols, asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            cov = returns_df[asset_symbols].cov().values
        else:
            cov = np.eye(n)

        if expected_returns is not None:
            mu = expected_returns.loc[asset_symbols].values
        elif returns_df is not None and not returns_df.empty:
            mu = returns_df[asset_symbols].mean().values * 252
        else:
            mu = np.ones(n) * 0.10

        def obj_func(w: np.ndarray) -> float:
            ret = float(np.dot(w, mu))
            var = float(np.dot(w.T, np.dot(cov, w)))
            return -(ret - 0.5 * self.risk_aversion * var)

        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        w0 = np.full(n, 1.0 / n)

        res = minimize(obj_func, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x if res.success else w0
        weights = np.maximum(0.0, weights)
        weights /= np.sum(weights)
        return {asset_symbols[i]: float(weights[i]) for i in range(n)}


class EqualRiskContributionModel(BaseAllocationModel):
    """Equal Risk Contribution (ERC) Capital Allocation Model."""

    def __init__(self) -> None:
        super().__init__("EqualRiskContribution")

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        return RiskParityModel().allocate(asset_symbols, returns_df, cov_matrix, expected_returns, views_dict)


class CustomAllocationModel(BaseAllocationModel):
    """Custom user-defined capital allocation model function."""

    def __init__(self, custom_fn: Callable[[List[str]], Dict[str, float]]) -> None:
        super().__init__("CustomAllocation")
        self.custom_fn = custom_fn

    def allocate(
        self,
        asset_symbols: List[str],
        returns_df: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        expected_returns: Optional[pd.Series] = None,
        views_dict: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        return self.custom_fn(asset_symbols)
