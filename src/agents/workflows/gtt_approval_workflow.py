"""LangGraph Human-in-the-Loop (HITL) GTT Approval Workflow.

Enforces strict separation:
1. Research Node (Technical Dip Scanner, Fundamentals, Corporate Events).
2. RiskGuard Node (Deterministic sizing, tick rounding, Clause 1.6, caps, circuit bounds).
3. Human Approval Node (LangGraph interrupt() barrier backed by durable SQLite checkpointer).
4. Execute GTT Node (quarantined strictly downstream of interrupt(); executes ONLY after human approval).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

import pandas as pd
import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from src.broker.gtt_executor import GTTExecutionResult, GTTExecutor, GTTOrderPayload
from src.config.settings import Settings, load_settings
from src.risk.risk_guard import (
    GTTStage,
    OrderGeometry,
    RiskGuard,
    RiskValidationRequest,
    RiskValidationResult,
    SizingResult,
)
from src.technicals.dip_scanner import TechnicalDipScanner

logger = structlog.get_logger(__name__)


def replace_value(current: Any, new: Any) -> Any:
    """Reducer that replaces previous value."""
    return new


class GTTWorkflowState(TypedDict):
    """Complete state schema for the HITL GTT Workflow."""

    symbol: Annotated[str, replace_value]
    query: Annotated[str, replace_value]
    stage: Annotated[int, replace_value]  # 1 (Buy) or 2 (Sell OCO)
    cmp: Annotated[float, replace_value]
    atr: Annotated[float, replace_value]
    portfolio_capital: Annotated[float, replace_value]
    available_cash: Annotated[float, replace_value]
    lower_circuit: Annotated[float, replace_value]
    upper_circuit: Annotated[float, replace_value]
    existing_holding_qty: Annotated[int, replace_value]
    existing_sector_value: Annotated[float, replace_value]
    sector: Annotated[str | None, replace_value]
    bse_events: Annotated[list[dict[str, Any]], replace_value]
    historical_bars: Annotated[Any, replace_value]  # pd.DataFrame
    thesis: Annotated[str | None, replace_value]
    fundamental_score: Annotated[int | None, replace_value]
    target_trigger: Annotated[float | None, replace_value]
    buy_trigger: Annotated[float | None, replace_value]
    old_trigger_id: Annotated[int | None, replace_value]  # For Place-Verify-Delete modification

    # Output / Intermediate State
    validation_result: Annotated[RiskValidationResult | None, replace_value]
    order_geometry: Annotated[OrderGeometry | None, replace_value]
    sizing_result: Annotated[SizingResult | None, replace_value]
    proposal_payload: Annotated[dict[str, Any] | None, replace_value]
    approval_decision: Annotated[dict[str, Any] | None, replace_value]
    execution_result: Annotated[GTTExecutionResult | None, replace_value]
    status: Annotated[str, replace_value]
    error: Annotated[str | None, replace_value]
    steps_completed: Annotated[list[str], operator.add]


# ── Graph Nodes ──────────────────────────────────────────────────────────────


async def research_node(
    state: GTTWorkflowState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Analyze dip candidates, scan indicators, and populate price targets if needed."""
    symbol = state.get("symbol", "").upper().strip()
    df: pd.DataFrame | None = state.get("historical_bars")

    updates: dict[str, Any] = {"steps_completed": ["research_node"]}

    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        scanner = TechnicalDipScanner()
        scan_res = scanner.evaluate(df, symbol=symbol)
        if not state.get("cmp"):
            updates["cmp"] = scan_res.current_close
        if not state.get("atr"):
            updates["atr"] = scan_res.atr14
        if not state.get("buy_trigger"):
            updates["buy_trigger"] = scan_res.entry_price
        if not state.get("target_trigger"):
            updates["target_trigger"] = scan_res.target
        if not state.get("thesis"):
            updates["thesis"] = (
                f"Technical dip on {symbol}: RSI(14)={scan_res.rsi14:.1f}, "
                f"ATR(14)=₹{scan_res.atr14:.2f}, GMMA={scan_res.gmma_trend}"
            )

    return updates


async def risk_guard_node(
    state: GTTWorkflowState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Validate candidate deterministically against statutory risk rules and position sizing."""
    symbol = state["symbol"].upper()
    stage = GTTStage.STAGE_1_BUY_DIP if state.get("stage", 1) == 1 else GTTStage.STAGE_2_SELL_OCO
    cmp = float(state.get("cmp", 0.0))
    atr = float(state.get("atr", cmp * 0.02))
    portfolio_capital = float(state.get("portfolio_capital", 1000000.0))
    available_cash = float(state.get("available_cash", portfolio_capital))
    lower_circuit = float(state.get("lower_circuit", cmp * 0.80))
    upper_circuit = float(state.get("upper_circuit", cmp * 1.20))
    corporate_events = state.get("bse_events") or []

    risk_guard = RiskGuard()
    req = RiskValidationRequest(
        tradingsymbol=symbol,
        stage=stage,
        cmp=cmp,
        atr=atr,
        portfolio_capital=portfolio_capital,
        available_cash=available_cash,
        lower_circuit_limit=lower_circuit,
        upper_circuit_limit=upper_circuit,
        buy_trigger=state.get("buy_trigger"),
        target_trigger=state.get("target_trigger"),
        requested_sell_qty=state.get("existing_holding_qty") if stage == GTTStage.STAGE_2_SELL_OCO else None,
        existing_holding_qty=state.get("existing_holding_qty", 0),
        existing_sector_value=state.get("existing_sector_value", 0.0),
        sector=state.get("sector"),
        corporate_events=corporate_events,
    )

    val_res = risk_guard.validate_and_size(req)
    if not val_res.is_valid:
        reasons = [r.value for r in val_res.rejection_reasons]
        details = "; ".join(val_res.rejection_details)
        logger.warning("risk_guard_rejected", symbol=symbol, reasons=reasons, details=details)
        return {
            "validation_result": val_res,
            "status": "REJECTED_BY_RISK_GUARD",
            "error": f"RiskGuard Rejected: {reasons} ({details})",
            "steps_completed": ["risk_guard_node"],
        }

    geom = val_res.geometry
    sizing = val_res.sizing
    gtt_payload = val_res.proposal_payload

    proposal_payload = {
        "tradingsymbol": symbol,
        "stage": state.get("stage", 1),
        "product": "CNC",
        "order_geometry": {
            "buy_trigger": geom.buy_trigger,
            "buy_limit": geom.buy_limit,
            "sl_trigger": geom.sl_trigger,
            "sl_limit": geom.sl_limit,
            "target_trigger": geom.target_trigger,
            "target_limit": geom.target_limit,
            "tick_size": geom.tick_size,
        },
        "sizing": {
            "quantity": sizing.quantity,
            "capital_deployed_inr": sizing.capital_deployed_inr,
            "capital_deployed_pct": sizing.capital_deployed_pct,
            "worst_case_risk_inr": sizing.worst_case_risk_inr,
            "worst_case_risk_pct": sizing.worst_case_risk_pct,
            "reward_risk_ratio": sizing.reward_risk_ratio_worst,
            "binding_constraint": sizing.binding_constraint,
        },
        "gtt_payload": gtt_payload,
        "thesis": state.get("thesis", f"Algorithmic dip buy on {symbol}"),
    }

    return {
        "validation_result": val_res,
        "order_geometry": geom,
        "sizing_result": sizing,
        "proposal_payload": proposal_payload,
        "status": "PENDING_APPROVAL",
        "error": None,
        "steps_completed": ["risk_guard_node"],
    }


def should_continue_after_risk_guard(
    state: GTTWorkflowState,
) -> Literal["human_gtt_approval_node", "rejection_node"]:
    """Conditional edge determining whether order passed RiskGuard."""
    if state.get("status") == "REJECTED_BY_RISK_GUARD" or not state.get("validation_result", {}).is_valid:
        return "rejection_node"
    return "human_gtt_approval_node"


async def human_gtt_approval_node(
    state: GTTWorkflowState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Human-in-the-Loop pause gate calling LangGraph interrupt().

    Guaranteed: NO MCP broker writes occur here.
    The graph pauses until resumed with Command(resume=decision).
    """
    proposal = state.get("proposal_payload") or {}
    logger.info("hitl_approval_gate_paused", symbol=state.get("symbol"))

    interrupt_payload = {
        "symbol": state.get("symbol"),
        "stage": state.get("stage", 1),
        "proposal": proposal,
        "thesis": state.get("thesis"),
        "metrics": proposal.get("sizing", {}),
    }

    # PAUSE GRAPH EXECUTION HERE
    user_decision = interrupt(interrupt_payload)

    logger.info("hitl_approval_gate_resumed", decision=user_decision)
    return {
        "approval_decision": user_decision,
        "steps_completed": ["human_gtt_approval_node"],
    }


async def execute_gtt_node(
    state: GTTWorkflowState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Execute GTT order strictly downstream of human approval.

    Resumed only when user submits decision via Command(resume=...).
    Handles Approve, Edit (re-validated with RiskGuard), and Reject.
    """
    decision = state.get("approval_decision") or {}
    action = decision.get("action", "reject").lower()
    symbol = state["symbol"].upper()
    stage = state.get("stage", 1)

    configurable = config.get("configurable", {})
    kite_client = configurable.get("kite_client")
    settings: Settings = configurable.get("settings") or load_settings()

    executor = GTTExecutor(
        kite_client=kite_client,
        settings=settings,
    )

    # 1. REJECT Branch
    if action == "reject":
        reason = decision.get("reason", "User rejected proposal in HITL review")
        executor.trade_logger.log_event(
            event_type="GTT_REJECTED",
            tradingsymbol=symbol,
            action="REJECT",
            stage=stage,
            dry_run=bool(settings.dry_run),
            details={"reason": reason},
        )
        executor.ledger.log_gtt_action(
            trigger_id=0,
            action="REJECT",
            stage=stage,
            tradingsymbol=symbol,
            trigger_price=0.0,
            limit_price=0.0,
            quantity=0,
            status="REJECTED_BY_USER",
            request_payload={"reason": reason},
        )
        res = GTTExecutionResult(
            success=True,
            action="REJECTED",
            stage=stage,
            tradingsymbol=symbol,
            dry_run=bool(settings.dry_run),
            status="REJECTED_BY_USER",
            message=f"Order proposal for {symbol} was rejected: {reason}",
            details={"reason": reason},
        )
        return {
            "execution_result": res,
            "status": "REJECTED_BY_USER",
            "steps_completed": ["execute_gtt_node"],
        }

    # 2. EDIT Branch (Re-validate overrides with RiskGuard)
    if action == "edit":
        overrides = decision.get("overrides", {})
        logger.info("hitl_edit_received", overrides=overrides)

        new_cmp = overrides.get("cmp", state.get("cmp"))
        new_buy_trigger = overrides.get("buy_trigger", state.get("buy_trigger"))
        new_target_trigger = overrides.get("target_trigger", state.get("target_trigger"))

        # Re-run RiskGuard validation
        risk_guard = RiskGuard()
        req = RiskValidationRequest(
            tradingsymbol=symbol,
            stage=GTTStage.STAGE_1_BUY_DIP if stage == 1 else GTTStage.STAGE_2_SELL_OCO,
            cmp=new_cmp,
            atr=state.get("atr", new_cmp * 0.02),
            portfolio_capital=state.get("portfolio_capital", 1000000.0),
            available_cash=state.get("available_cash", 1000000.0),
            lower_circuit_limit=state.get("lower_circuit", new_cmp * 0.80),
            upper_circuit_limit=state.get("upper_circuit", new_cmp * 1.20),
            buy_trigger=new_buy_trigger,
            target_trigger=new_target_trigger,
            existing_holding_qty=state.get("existing_holding_qty", 0),
            existing_sector_value=state.get("existing_sector_value", 0.0),
            sector=state.get("sector"),
            corporate_events=state.get("bse_events") or [],
        )

        val_res = risk_guard.validate_and_size(req)
        if not val_res.is_valid:
            reasons = [r.value for r in val_res.rejection_reasons]
            return {
                "validation_result": val_res,
                "status": "EDIT_REJECTED_BY_RISK_GUARD",
                "error": f"Edited parameters failed RiskGuard: {reasons}",
                "steps_completed": ["execute_gtt_node"],
            }

        # Override geometry & sizing
        geom = val_res.geometry
        sizing = val_res.sizing
    else:
        # 3. APPROVE Branch (Use validated geometry)
        geom = state["order_geometry"]
        sizing = state["sizing_result"]

    # Construct order payload
    if stage == 1:
        # Stage 1 Single-Leg Buy
        order_payload = GTTOrderPayload(
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type="BUY",
            trigger_type="single",
            trigger_values=[geom.buy_trigger],
            limit_prices=[geom.buy_limit],
            quantities=[sizing.quantity],
            product="CNC",
            stage=1,
            thesis=state.get("thesis"),
            allocated_cash=sizing.capital_deployed_inr,
        )
    else:
        # Stage 2 Two-Leg Sell OCO
        order_payload = GTTOrderPayload(
            tradingsymbol=symbol,
            exchange="NSE",
            transaction_type="SELL",
            trigger_type="two-leg",
            trigger_values=[geom.sl_trigger, geom.target_trigger],
            limit_prices=[geom.sl_limit, geom.target_limit],
            quantities=[sizing.quantity, sizing.quantity],
            product="CNC",
            stage=2,
            thesis=state.get("thesis"),
        )

    # Place or Place-Verify-Delete
    old_id = state.get("old_trigger_id")
    if old_id:
        res = await executor.modify_gtt_place_verify_delete(old_id, order_payload)
    else:
        res = await executor.place_gtt(order_payload)

    return {
        "execution_result": res,
        "status": "APPROVED_AND_EXECUTED",
        "error": None,
        "steps_completed": ["execute_gtt_node"],
    }


async def rejection_node(
    state: GTTWorkflowState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Terminal node reached when RiskGuard rejects proposal prior to approval."""
    logger.info("workflow_ended_at_rejection", symbol=state.get("symbol"), status=state.get("status"))
    return {
        "steps_completed": ["rejection_node"],
    }


# ── Workflow Graph Builder ────────────────────────────────────────────────────


def create_gtt_approval_workflow() -> StateGraph:
    """Construct the state graph for the HITL GTT Approval Workflow."""
    builder = StateGraph(GTTWorkflowState)

    builder.add_node("research_node", research_node)
    builder.add_node("risk_guard_node", risk_guard_node)
    builder.add_node("human_gtt_approval_node", human_gtt_approval_node)
    builder.add_node("execute_gtt_node", execute_gtt_node)
    builder.add_node("rejection_node", rejection_node)

    # Wire edges
    builder.set_entry_point("research_node")
    builder.add_edge("research_node", "risk_guard_node")
    builder.add_conditional_edges(
        "risk_guard_node",
        should_continue_after_risk_guard,
        {
            "human_gtt_approval_node": "human_gtt_approval_node",
            "rejection_node": "rejection_node",
        },
    )
    builder.add_edge("human_gtt_approval_node", "execute_gtt_node")
    builder.add_edge("execute_gtt_node", END)
    builder.add_edge("rejection_node", END)

    return builder
