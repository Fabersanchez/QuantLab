"""Trend Indicators Package."""

from indicators.categories.trend.sma import SMAIndicator
from indicators.categories.trend.ema import EMAIndicator
from indicators.categories.trend.supertrend import SuperTrendIndicator

__all__ = ["SMAIndicator", "EMAIndicator", "SuperTrendIndicator"]
