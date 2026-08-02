"""QuantLab Indicator Categories Package."""

from indicators.categories.trend import SMAIndicator, EMAIndicator, SuperTrendIndicator
from indicators.categories.momentum import RSIIndicator, MACDIndicator
from indicators.categories.volatility import ATRIndicator, BollingerBandsIndicator
from indicators.categories.volume import OBVIndicator, VWAPIndicator
from indicators.categories.price_action import PinBarIndicator, EngulfingPatternIndicator
from indicators.categories.market_structure import SwingPointsIndicator, BOSCHoCHIndicator
from indicators.categories.smart_money import FairValueGapIndicator, OrderBlocksIndicator
from indicators.categories.liquidity import LiquidityPoolsIndicator
from indicators.categories.order_flow import CVDIndicator
from indicators.categories.statistical import ZScoreIndicator
from indicators.categories.cycle import HilbertTransformIndicator
from indicators.categories.pattern import DoubleTopBottomIndicator

ALL_BUILTIN_INDICATORS = [
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
    BOSCHoCHIndicator,
    FairValueGapIndicator,
    OrderBlocksIndicator,
    LiquidityPoolsIndicator,
    CVDIndicator,
    ZScoreIndicator,
    HilbertTransformIndicator,
    DoubleTopBottomIndicator,
]
