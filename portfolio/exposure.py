"""
QuantLab Portfolio Exposure & Leverage Analyzer.

Calculates Long exposure, Short exposure, Net exposure, Gross exposure,
Leverage ratios, and Sector / Asset Class risk exposure breakdowns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from portfolio.portfolio import Portfolio


@dataclass
class PortfolioExposure:
    """Dataclass holding detailed portfolio exposure and leverage metrics."""

    long_exposure_val: float = 0.0
    short_exposure_val: float = 0.0
    net_exposure_val: float = 0.0
    gross_exposure_val: float = 0.0
    long_exposure_pct: float = 0.0
    short_exposure_pct: float = 0.0
    net_exposure_pct: float = 0.0
    gross_exposure_pct: float = 0.0
    leverage_ratio: float = 1.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    market_exposure: Dict[str, float] = field(default_factory=dict)


class ExposureAnalyzer:
    """Institutional Portfolio Exposure Analyzer."""

    @staticmethod
    def analyze(portfolio: Portfolio, position_values: Dict[str, float]) -> PortfolioExposure:
        """Analyze portfolio positions to calculate long, short, net, gross exposure and leverage.

        Args:
            portfolio: Portfolio instance.
            position_values: Dictionary mapping asset symbol to signed position dollar value.

        Returns:
            PortfolioExposure instance.
        """
        capital = float(portfolio.initial_capital) or 100000.0

        long_val = 0.0
        short_val = 0.0
        sector_map: Dict[str, float] = {}
        market_map: Dict[str, float] = {}

        for sym, val in position_values.items():
            if val >= 0:
                long_val += val
            else:
                short_val += abs(val)

            asset = portfolio.assets.get(sym)
            sec = asset.sector if asset else "Uncategorized"
            mkt = asset.market.value if (asset and hasattr(asset.market, "value")) else "Other"

            sector_map[sec] = sector_map.get(sec, 0.0) + abs(val)
            market_map[mkt] = market_map.get(mkt, 0.0) + abs(val)

        net_val = long_val - short_val
        gross_val = long_val + short_val

        long_pct = (long_val / capital) * 100.0
        short_pct = (short_val / capital) * 100.0
        net_pct = (net_val / capital) * 100.0
        gross_pct = (gross_val / capital) * 100.0
        leverage = gross_val / capital

        return PortfolioExposure(
            long_exposure_val=long_val,
            short_exposure_val=short_val,
            net_exposure_val=net_val,
            gross_exposure_val=gross_val,
            long_exposure_pct=long_pct,
            short_exposure_pct=short_pct,
            net_exposure_pct=net_pct,
            gross_exposure_pct=gross_pct,
            leverage_ratio=leverage,
            sector_exposure=sector_map,
            market_exposure=market_map,
        )
