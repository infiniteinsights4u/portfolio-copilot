"""Technical Dip Agent LangGraph workflow.

Evaluates candidates using clean-room indicators:
- Wilder's RSI(14) pullback-curl (30-42)
- 14-day Wilder ATR for 2.0x ATR stop-loss sizing
- 50/200-DMA trend filter
- GMMA Guppy Multiple Moving Average ribbons
- Heikin-Ashi candlestick swing filter
- 52-week drawdown and 20-day delivery % accumulation check
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pandas as pd
import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from src.agents.tools.symbol_tools import extract_symbol
from src.technicals.dip_scanner import DipScanResult, TechnicalDipScanner

logger = structlog.get_logger(__name__)


def replace_value(current: Any, new: Any) -> Any:
    """Reducer that replaces the current value with the new one."""
    return new


class TechnicalDipState(TypedDict):
    """State schema for the Technical Dip Agent workflow."""

    query: str
    symbol: Annotated[str, replace_value]
    historical_bars: Annotated[Any, replace_value]  # pd.DataFrame
    dip_scan_result: Annotated[DipScanResult | None, replace_value]
    is_valid_dip: Annotated[bool, replace_value]
    proposed_entry: Annotated[float, replace_value]
    proposed_stop_loss: Annotated[float, replace_value]
    proposed_target: Annotated[float, replace_value]
    risk_reward_ratio: Annotated[float, replace_value]
    reasons: Annotated[list[str], replace_value]
    error: Annotated[str | None, replace_value]
    steps_completed: Annotated[list, operator.add]


async def extract_symbol_node(
    state: TechnicalDipState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Node: Extract or standardize stock symbol."""
    symbol = state.get("symbol", "").strip()
    if not symbol:
        query = state.get("query", "")
        extracted = extract_symbol(query)
        if extracted:
            symbol = extracted

    if not symbol:
        return {
            "error": "No stock symbol provided or detected for technical scan.",
            "steps_completed": ["extract_symbol"],
        }

    return {
        "symbol": symbol.upper(),
        "error": None,
        "steps_completed": ["extract_symbol"],
    }


async def evaluate_technical_dip_node(
    state: TechnicalDipState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Node: Compute indicators and scan for valid CNC dip setups."""
    symbol = state.get("symbol", "").upper()
    df: pd.DataFrame | None = state.get("historical_bars")

    # If df not in state, check if provided in config configurable
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        configurable = config.get("configurable", {})
        df = configurable.get("historical_bars")

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("technical_scan_missing_bars", symbol=symbol)
        return {
            "error": f"No historical price bars available for symbol {symbol}",
            "is_valid_dip": False,
            "steps_completed": ["evaluate_technical_dip"],
        }

    scanner = TechnicalDipScanner()
    result = scanner.evaluate(df, symbol=symbol)

    return {
        "dip_scan_result": result,
        "is_valid_dip": result.is_valid_dip,
        "proposed_entry": result.suggested_entry,
        "proposed_stop_loss": result.suggested_stop_loss,
        "proposed_target": result.suggested_target,
        "risk_reward_ratio": result.risk_reward_ratio,
        "reasons": result.reasons,
        "error": None,
        "steps_completed": ["evaluate_technical_dip"],
    }


def create_technical_dip_graph() -> StateGraph:
    """Build and compile the Technical Dip LangGraph workflow."""
    workflow = StateGraph(TechnicalDipState)

    workflow.add_node("extract_symbol", extract_symbol_node)
    workflow.add_node("evaluate_technical_dip", evaluate_technical_dip_node)

    workflow.set_entry_point("extract_symbol")

    def route_after_extract(state: TechnicalDipState) -> str:
        if state.get("error"):
            return END
        return "evaluate_technical_dip"

    workflow.add_conditional_edges("extract_symbol", route_after_extract)
    workflow.add_edge("evaluate_technical_dip", END)

    return workflow.compile()
