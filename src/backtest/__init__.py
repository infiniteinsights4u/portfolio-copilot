"""Split-adjusted EOD backtesting engine for Zerodha CNC dip-investing."""

from src.backtest.eod_engine import (
    BacktestConfig,
    BacktestResult,
    CostBreakdown,
    EODBacktestEngine,
    TradeRecord,
    TransactionCostModel,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "CostBreakdown",
    "EODBacktestEngine",
    "TradeRecord",
    "TransactionCostModel",
]
