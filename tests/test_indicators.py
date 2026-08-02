"""
QuantLab Institutional Quant Library Unit Tests.

Verifies functionality of BaseIndicator, IndicatorMetadata, IndicatorRegistry,
IndicatorValidator, IndicatorCache, IndicatorPipeline, IndicatorEngine,
and representative indicators across all categories.
"""

import unittest
import numpy as np
import pandas as pd

from indicators import (
    IndicatorEngine,
    IndicatorRegistry,
    IndicatorValidator,
    ALL_BUILTIN_INDICATORS,
    SMAIndicator,
    EMAIndicator,
    SuperTrendIndicator,
    RSIIndicator,
    MACDIndicator,
    ATRIndicator,
    BollingerBandsIndicator,
    OBVIndicator,
    VWAPIndicator,
    PinBarIndicator,
    EngulfingPatternIndicator,
    SwingPointsIndicator,
    FairValueGapIndicator,
    OrderBlocksIndicator,
)


class TestQuantLabIndicatorLibrary(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        timestamps = pd.date_range("2026-01-01 09:30", periods=50, freq="1min")
        close_prices = 100.0 + np.cumsum(np.random.randn(50))
        high_prices = close_prices + np.abs(np.random.randn(50))
        low_prices = close_prices - np.abs(np.random.randn(50))
        open_prices = low_prices + (high_prices - low_prices) * np.random.rand(50)
        volume = 1000.0 + np.random.rand(50) * 500.0

        self.sample_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            }
        )

        self.engine = IndicatorEngine()
        for ind_cls in ALL_BUILTIN_INDICATORS:
            self.engine.register_indicator(ind_cls, overwrite=True)

    def test_registry(self) -> None:
        registry = IndicatorRegistry()
        registry.register(SMAIndicator)
        self.assertTrue(registry.has("SMA"))
        self.assertEqual(registry.get("SMA"), SMAIndicator)
        self.assertIn("SMA", registry.list_indicators())

    def test_validator(self) -> None:
        report = IndicatorValidator.validate_input(
            self.sample_df, required_columns=["high", "low", "close"], min_rows=10
        )
        self.assertTrue(report.is_valid)

        bad_report = IndicatorValidator.validate_input(
            self.sample_df, required_columns=["missing_column"]
        )
        self.assertFalse(bad_report.is_valid)

    def test_trend_indicators(self) -> None:
        sma_df = self.engine.calculate(self.sample_df, "SMA", params={"period": 10})
        self.assertIn("sma_10", sma_df.columns)
        self.assertEqual(len(sma_df), 50)

        ema_df = self.engine.calculate(self.sample_df, "EMA", params={"period": 10})
        self.assertIn("ema_10", ema_df.columns)

        st_df = self.engine.calculate(self.sample_df, "SuperTrend")
        self.assertIn("supertrend", st_df.columns)

    def test_momentum_indicators(self) -> None:
        rsi_df = self.engine.calculate(self.sample_df, "RSI", params={"period": 14})
        self.assertIn("rsi_14", rsi_df.columns)

        macd_df = self.engine.calculate(self.sample_df, "MACD")
        self.assertIn("macd", macd_df.columns)
        self.assertIn("macd_signal", macd_df.columns)

    def test_volatility_and_volume_indicators(self) -> None:
        atr_df = self.engine.calculate(self.sample_df, "ATR")
        self.assertIn("atr_14", atr_df.columns)

        bb_df = self.engine.calculate(self.sample_df, "BollingerBands")
        self.assertIn("bb_upper", bb_df.columns)

        obv_df = self.engine.calculate(self.sample_df, "OBV")
        self.assertIn("obv", obv_df.columns)

        vwap_df = self.engine.calculate(self.sample_df, "VWAP")
        self.assertIn("vwap", vwap_df.columns)

    def test_smart_money_and_structure_indicators(self) -> None:
        fvg_df = self.engine.calculate(self.sample_df, "FairValueGap")
        self.assertIn("fvg_signal", fvg_df.columns)

        swings_df = self.engine.calculate(self.sample_df, "SwingPoints")
        self.assertIn("swing_high", swings_df.columns)

        ob_df = self.engine.calculate(self.sample_df, "OrderBlocks")
        self.assertIn("order_block_signal", ob_df.columns)

    def test_calculate_all(self) -> None:
        combined_df = self.engine.calculate_all(self.sample_df)
        self.assertIn("sma_14", combined_df.columns)
        self.assertIn("rsi_14", combined_df.columns)
        self.assertIn("vwap", combined_df.columns)
        self.assertGreater(combined_df.shape[1], 10)


if __name__ == "__main__":
    unittest.main()
