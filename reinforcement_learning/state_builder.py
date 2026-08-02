"""
QuantLab Reinforcement Learning - State & Observation Vector Builder.

Constructs normalized observation vectors for RL agents from OHLCV price history,
technical indicators, market regime labels, current portfolio equity, open positions,
entry prices, and current drawdown percentage.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Snapshot of current agent portfolio state."""

    equity: float = 10_000.0
    cash: float = 10_000.0
    position: int = 0          # -1 Short, 0 Flat, +1 Long
    position_size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    peak_equity: float = 10_000.0
    drawdown_pct: float = 0.0
    n_trades: int = 0


class StateBuilder:
    """Constructs normalized RL observation vectors for financial market environments.

    Observation vector composition (configurable):
        - OHLCV window: shape (lookback, 5)
        - Feature/Indicator window: shape (lookback, n_features)
        - Market regime: shape (1,)         (0=Bear, 0.5=Sideways, 1=Bull)
        - Portfolio stats: shape (7,)       (equity, cash, position, size, unrealized_pnl, drawdown, n_trades)

    All values are normalized to zero-mean unit-variance or [0, 1] range.
    """

    def __init__(
        self,
        lookback: int = 20,
        feature_cols: Optional[List[str]] = None,
        ohlcv_cols: Optional[List[str]] = None,
        include_portfolio: bool = True,
        include_regime: bool = True,
    ) -> None:
        """Initialize StateBuilder.

        Args:
            lookback: Number of past OHLCV bars to include in the observation.
            feature_cols: List of indicator/feature column names to include.
            ohlcv_cols: OHLCV column names. Defaults to ['open','high','low','close','volume'].
            include_portfolio: Whether to append portfolio state to the observation.
            include_regime: Whether to append market regime indicator.
        """
        self.lookback = lookback
        self.feature_cols: List[str] = feature_cols or []
        self.ohlcv_cols: List[str] = ohlcv_cols or ["open", "high", "low", "close", "volume"]
        self.include_portfolio = include_portfolio
        self.include_regime = include_regime

        # Observation dim = (lookback * (5 + n_features)) + optional_extras
        n_ohlcv = len(self.ohlcv_cols) * self.lookback
        n_feat = len(self.feature_cols) * self.lookback
        n_portfolio = 7 if include_portfolio else 0
        n_regime = 1 if include_regime else 0
        self._obs_dim: int = n_ohlcv + n_feat + n_portfolio + n_regime

    @property
    def obs_dim(self) -> int:
        """Return flat observation dimension size."""
        return self._obs_dim

    def build(
        self,
        df: pd.DataFrame,
        current_idx: int,
        portfolio: Optional[PortfolioState] = None,
        market_regime: float = 0.5,
    ) -> np.ndarray:
        """Build a normalized flat observation vector for the agent.

        Args:
            df: Full market DataFrame (OHLCV + features).
            current_idx: Current bar index in the DataFrame.
            portfolio: Current PortfolioState snapshot.
            market_regime: Normalized market regime value (0=Bear, 0.5=Sideways, 1=Bull).

        Returns:
            Flat numpy observation vector of shape (obs_dim,).
        """
        start = max(0, current_idx - self.lookback + 1)
        end = current_idx + 1
        window = df.iloc[start:end]

        # Pad if window is shorter than lookback
        pad_len = self.lookback - len(window)

        obs_parts: List[np.ndarray] = []

        # OHLCV block
        ohlcv_data = window[self.ohlcv_cols].values.astype(np.float32)
        if pad_len > 0:
            ohlcv_data = np.vstack([np.zeros((pad_len, len(self.ohlcv_cols)), dtype=np.float32), ohlcv_data])
        obs_parts.append(self._normalize(ohlcv_data.flatten()))

        # Feature / indicator block
        if self.feature_cols:
            available = [c for c in self.feature_cols if c in window.columns]
            feat_data = window[available].values.astype(np.float32) if available else np.zeros((len(window), len(self.feature_cols)), dtype=np.float32)
            if pad_len > 0:
                feat_data = np.vstack([np.zeros((pad_len, feat_data.shape[1]), dtype=np.float32), feat_data])
            obs_parts.append(self._normalize(feat_data.flatten()))

        # Market regime
        if self.include_regime:
            obs_parts.append(np.array([float(market_regime)], dtype=np.float32))

        # Portfolio state vector
        if self.include_portfolio:
            if portfolio is None:
                portfolio = PortfolioState()
            port_vec = np.array([
                portfolio.equity / 10_000.0,
                portfolio.cash / 10_000.0,
                float(portfolio.position),
                portfolio.position_size,
                portfolio.unrealized_pnl / max(1.0, portfolio.equity),
                portfolio.drawdown_pct,
                float(portfolio.n_trades) / 100.0,
            ], dtype=np.float32)
            obs_parts.append(port_vec)

        obs = np.concatenate(obs_parts)

        # Safety: ensure correct dimension
        if len(obs) < self._obs_dim:
            obs = np.pad(obs, (0, self._obs_dim - len(obs)))
        elif len(obs) > self._obs_dim:
            obs = obs[:self._obs_dim]

        return obs.astype(np.float32)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize array to zero-mean unit-variance (safe for constant arrays)."""
        std = arr.std()
        if std < 1e-8:
            return arr - arr.mean()
        return (arr - arr.mean()) / std
