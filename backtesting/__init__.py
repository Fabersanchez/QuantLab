"""
QuantLab Institutional Backtesting Engine Package.

Provides realistic quantitative backtesting, execution simulation, commission/slippage/spread/latency models,
order and position lifecycle management, portfolio margin and risk accounting, analytics,
multi-format reporting, and step-by-step simulation replay.
"""

from backtesting.commission_model import (
    BaseCommissionModel,
    FixedCommissionModel,
    PercentageCommissionModel,
    PerLotCommissionModel,
    BrokerSpecificCommissionModel,
    CustomCommissionModel,
    CommissionModelFactory,
)
from backtesting.slippage_model import (
    BaseSlippageModel,
    FixedSlippageModel,
    DynamicSlippageModel,
    VolatilityBasedSlippageModel,
    LiquidityBasedSlippageModel,
    RandomSlippageModel,
    SlippageModelFactory,
)
from backtesting.spread_model import (
    BaseSpreadModel,
    FixedSpreadModel,
    VariableSpreadModel,
    HistoricalSpreadModel,
    BrokerSpreadModel,
    SpreadModelFactory,
)
from backtesting.latency_model import (
    BaseLatencyModel,
    ExecutionDelayModel,
    NetworkDelayModel,
    ExchangeDelayModel,
    BrokerDelayModel,
    CompositeLatencyModel,
    LatencyModelFactory,
)
from backtesting.order_manager import (
    Order,
    OrderManager,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)
from backtesting.position_manager import (
    Position,
    PositionManager,
    PositionSide,
    ScalingEvent,
)
from backtesting.portfolio_manager import (
    PortfolioManager,
    PortfolioState,
)
from backtesting.execution_simulator import ExecutionSimulator
from backtesting.equity_curve import EquityCurve, EquityPoint
from backtesting.trade_log import TradeLog, TradeRecord
from backtesting.metrics import PerformanceMetrics
from backtesting.statistics import BacktestStatistics
from backtesting.report_generator import ReportGenerator
from backtesting.replay import BacktestReplay, ReplaySnapshot
from backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from backtesting.backtest_runner import BacktestRunner

__all__ = [
    # Commission Models
    "BaseCommissionModel",
    "FixedCommissionModel",
    "PercentageCommissionModel",
    "PerLotCommissionModel",
    "BrokerSpecificCommissionModel",
    "CustomCommissionModel",
    "CommissionModelFactory",
    # Slippage Models
    "BaseSlippageModel",
    "FixedSlippageModel",
    "DynamicSlippageModel",
    "VolatilityBasedSlippageModel",
    "LiquidityBasedSlippageModel",
    "RandomSlippageModel",
    "SlippageModelFactory",
    # Spread Models
    "BaseSpreadModel",
    "FixedSpreadModel",
    "VariableSpreadModel",
    "HistoricalSpreadModel",
    "BrokerSpreadModel",
    "SpreadModelFactory",
    # Latency Models
    "BaseLatencyModel",
    "ExecutionDelayModel",
    "NetworkDelayModel",
    "ExchangeDelayModel",
    "BrokerDelayModel",
    "CompositeLatencyModel",
    "LatencyModelFactory",
    # Order & Position Managers
    "Order",
    "OrderManager",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "Position",
    "PositionManager",
    "PositionSide",
    "ScalingEvent",
    # Portfolio & Execution
    "PortfolioManager",
    "PortfolioState",
    "ExecutionSimulator",
    "EquityCurve",
    "EquityPoint",
    "TradeLog",
    "TradeRecord",
    # Analytics & Reports
    "PerformanceMetrics",
    "BacktestStatistics",
    "ReportGenerator",
    "BacktestReplay",
    "ReplaySnapshot",
    # Engine & Runner
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunner",
]
