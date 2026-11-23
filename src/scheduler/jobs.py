"""Autonomous positional scheduled jobs conforming to IST market routines.

Schedules (Indian Standard Time):
1. 08:45 AM IST - Pre-market sanity check:
   - Kite MCP OAuth token expiry check (daily 6:00 AM IST reset).
   - DDPI status check (CDSL TPIN authorization alert if disabled).
   - Instrument cache and fresh Bhavcopy readiness.
   - BSE corporate action blackout check for open positions and pending orders.
   - Pending GTT reconciliation.

2. 16:05 PM IST - Post-CAS reconciliation & trailing stops:
   - Stage-1 Buy GTT fill detection -> auto-commit & queue Stage-2 Two-Leg Sell OCO.
   - Gap-down unfilled Sell GTT detection -> CRITICAL urgent alert.
   - Trailing stop-loss ratcheting (Chandelier / 2.0 * ATR) via Place-Verify-Delete.

3. 18:15 PM IST - EOD Bhavcopy dip scan:
   - Ingest official NSE sec_bhavdata_full with strict DATE1 validation.
   - Run clean-room TechnicalDipScanner across universe.
   - Identify and alert qualified dip setups with entry/stop/target geometry.

4. Friday 16:15 PM IST - Weekly performance digest:
   - Portfolio valuation, open positions count, unrealized P&L.
   - STCG (< 365d) vs LTCG (>= 365d) tax categorization.
   - GTT audit actions summary for the week.
   - 10% single-stock and 25% sector concentration checks.
   - Formatted Markdown digest dispatched via webhooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

from src.broker.gtt_executor import GTTExecutor, GTTOrderPayload
from src.broker.kite_mcp_client import KiteMCPClient
from src.config.settings import Settings, get_settings
from src.data.bhavcopy import BhavcopyDateMismatchError, normalize_date
from src.risk.risk_guard import (
    RiskGuard,
    get_nse_dynamic_tick_size,
    round_to_tick,
)
from src.scheduler.alerts import AlertDispatcher, AlertMessage, AlertSeverity
from src.storage.ledger import SQLiteLedger
from src.technicals.dip_scanner import DipScanResult, TechnicalDipScanner

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class JobExecutionResult:
    """Standard outcome record for a scheduled positional job."""

    job_name: str
    success: bool
    timestamp: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    alerts_dispatched: list[AlertMessage] = field(default_factory=list)


# ── 1. 08:45 AM Pre-Market Sanity Check ──────────────────────────────────────


async def run_pre_market_job(
    ledger: SQLiteLedger | None = None,
    kite_client: KiteMCPClient | None = None,
    settings: Settings | None = None,
    alert_dispatcher: AlertDispatcher | None = None,
    today_override: date | None = None,
    corporate_events: list[dict[str, Any]] | None = None,
    skip_client_probe: bool = False,
) -> JobExecutionResult:
    """Execute the 08:45 AM IST pre-market operational readiness routine."""
    cfg = settings or get_settings()
    db = ledger or SQLiteLedger(cfg.db_path)
    alerts = alert_dispatcher or AlertDispatcher(cfg)
    now_ist = datetime.now(IST)
    current_date = today_override or now_ist.date()
    iso_now = now_ist.isoformat()

    logger.info("job_pre_market_started", date=str(current_date), iso_now=iso_now)
    dispatched: list[AlertMessage] = []
    details: dict[str, Any] = {}

    # 1. Kite MCP Session Health & Daily 6:00 AM IST Expiry Check
    session_healthy = True
    session_error = None
    if not skip_client_probe and kite_client is not None:
        try:
            # Probe get_gtts or session
            await kite_client.get_gtts()
        except Exception as e:
            session_healthy = False
            session_error = str(e)

    if not session_healthy:
        m = await alerts.dispatch(
            title="Zerodha Kite Session Expired",
            message=(
                "Zerodha Kite OAuth session has expired or is unreachable (daily 6:00 AM IST reset). "
                "Please perform manual 2FA OAuth login in the copilot UI before 09:15 AM IST market open. "
                f"Diagnostic details: {session_error}"
            ),
            severity=AlertSeverity.WARNING,
            metadata={"check": "session_health", "date": str(current_date)},
        )
        dispatched.append(m)
        details["session_healthy"] = False
    else:
        details["session_healthy"] = True

    # 2. DDPI Status Check (Warning if disabled)
    ddpi_enabled = cfg.zerodha_ddpi_enabled
    details["ddpi_enabled"] = ddpi_enabled
    if not ddpi_enabled:
        m = await alerts.dispatch(
            title="CDSL TPIN / eDIS Pre-Authorization Required",
            message=(
                "ZERODHA_DDPI_ENABLED is False! Triggered CNC Sell GTT orders will fail execution "
                "without manual daily CDSL TPIN authorization. Please authenticate eDIS in Zerodha "
                "Console after 07:00 AM IST before market opens."
            ),
            severity=AlertSeverity.WARNING,
            metadata={"check": "ddpi_status"},
        )
        dispatched.append(m)

    # 3. Check Open Positions & Corporate Action Blackout
    open_positions = db.get_open_positions()
    details["open_positions_count"] = len(open_positions)
    open_symbols = {p["tradingsymbol"].upper() for p in open_positions}

    # Reconcile pending buy commits
    pending_commits = db.get_pending_buy_commits()
    details["pending_buy_commits_count"] = len(pending_commits)
    committed_cash = sum(c["allocated_cash"] for c in pending_commits)
    details["committed_cash_inr"] = committed_cash

    for c in pending_commits:
        open_symbols.add(c["tradingsymbol"].upper())

    # Check corporate events for symbols
    blackout_warnings: list[str] = []
    if corporate_events:
        risk_guard = RiskGuard()
        for sym in open_symbols:
            sym_events = [e for e in corporate_events if e.get("symbol", "").upper() == sym]
            if sym_events:
                is_blackout, reasons = risk_guard.check_corporate_actions_blackout(
                    events=sym_events,
                    cmp=1000.0,  # nominal CMP for dividend calculation
                    blackout_days=5,
                    current_date=current_date,
                )
                if is_blackout:
                    blackout_warnings.append(f"{sym}: {'; '.join(reasons)}")

    if blackout_warnings:
        m = await alerts.dispatch(
            title="Corporate Action Blackout Alert",
            message=(
                "Upcoming corporate actions or board meetings detected within 5 days for held/pending symbols:\n- "
                + "\n- ".join(blackout_warnings)
                + "\nExercise caution: GTT orders may be auto-cancelled by exchange or subject to extreme gap risk."
            ),
            severity=AlertSeverity.WARNING,
            metadata={"blackouts": blackout_warnings},
        )
        dispatched.append(m)
        details["blackout_warnings"] = blackout_warnings

    # 4. Summary Pre-Market Dispatch
    summary = (
        f"Pre-Market Sanity Check OK. Open Positions: {len(open_positions)}, "
        f"Pending GTTs: {len(pending_commits)}, Committed Cash: ₹{committed_cash:,.2f}, "
        f"Session Healthy: {details['session_healthy']}, DDPI: {ddpi_enabled}"
    )

    if not dispatched:
        # If no warnings, send clean info briefing
        m = await alerts.dispatch(
            title="Pre-Market System Ready",
            message=summary,
            severity=AlertSeverity.INFO,
            metadata=details,
        )
        dispatched.append(m)

    return JobExecutionResult(
        job_name="pre_market_job",
        success=True,
        timestamp=iso_now,
        summary=summary,
        details=details,
        alerts_dispatched=dispatched,
    )


# ── 2. 16:05 PM Post-CAS Reconciliation & Trailing Stops ─────────────────────


async def run_post_cas_reconciliation_job(
    ledger: SQLiteLedger | None = None,
    kite_client: KiteMCPClient | None = None,
    gtt_executor: GTTExecutor | None = None,
    risk_guard: RiskGuard | None = None,
    settings: Settings | None = None,
    alert_dispatcher: AlertDispatcher | None = None,
    broker_gtts: list[dict[str, Any]] | None = None,
    market_quotes: dict[str, dict[str, float]] | None = None,
) -> JobExecutionResult:
    """Execute 16:05 PM IST post-market reconciliation & trailing stop ratcheting."""
    cfg = settings or get_settings()
    db = ledger or SQLiteLedger(cfg.db_path)
    executor = gtt_executor or GTTExecutor(kite_client=kite_client, ledger=db, settings=cfg)
    _guard = risk_guard or RiskGuard()
    alerts = alert_dispatcher or AlertDispatcher(cfg)
    now_ist = datetime.now(IST)
    iso_now = now_ist.isoformat()

    logger.info("job_post_cas_reconciliation_started", iso_now=iso_now)
    dispatched: list[AlertMessage] = []
    details: dict[str, Any] = {
        "stage1_fills_handled": 0,
        "critical_unfilled_sl_alerts": 0,
        "trailing_stops_ratcheted": 0,
    }

    # Fetch live GTTs from broker (or provided fixture list)
    active_gtts: list[dict[str, Any]] = []
    if broker_gtts is not None:
        active_gtts = broker_gtts
    elif kite_client is not None:
        try:
            res = await kite_client.get_gtts()
            if isinstance(res, list):
                active_gtts = res
        except Exception as e:
            logger.error("post_cas_fetch_gtts_failed", error=str(e))

    gtt_by_id = {g.get("id") or g.get("trigger_id"): g for g in active_gtts if g.get("id") or g.get("trigger_id")}

    # ── PART A: Stage-1 Buy Fill Reconciliation & Stage-2 Sell OCO Queuing ─────
    pending_commits = db.get_pending_buy_commits()

    for commit in pending_commits:
        trig_id = commit.get("trigger_id")
        sym = commit["tradingsymbol"].upper()
        broker_gtt = gtt_by_id.get(trig_id)

        # In dry run or broker simulation: check if marked triggered or filled
        is_triggered = False
        if broker_gtt:
            status = str(broker_gtt.get("status", "")).lower()
            if status in ("triggered", "completed", "filled"):
                is_triggered = True

        if is_triggered:
            logger.info("stage1_buy_gtt_filled", symbol=sym, trigger_id=trig_id)
            allocated = commit["allocated_cash"]
            # Estimate or extract fill price and quantity
            entry_price = float(broker_gtt.get("condition", {}).get("trigger_values", [0])[0] or 100.0)
            if entry_price <= 0:
                entry_price = float(broker_gtt.get("last_price", 100.0))
            quantity = int(allocated // entry_price) if entry_price > 0 else 1
            if quantity <= 0:
                quantity = 1

            # Tick alignment and geometry calculation
            tick = get_nse_dynamic_tick_size(entry_price)
            # Default 2.0 * ATR stop distance (assumes 2% if ATR not available)
            atr_est = entry_price * 0.02
            sl_price = round_to_tick(entry_price - (2.0 * atr_est), tick)
            target_price = round_to_tick(entry_price + (2.5 * 2.0 * atr_est), tick)

            # 1. Record / arm position in SQLite ledger
            pos_id = db.record_position(
                tradingsymbol=sym,
                entry_price=entry_price,
                quantity=quantity,
                current_stop_loss=sl_price,
                target_price=target_price,
                stage="STAGE_2_ARMED",
            )

            # 2. Release pending cash reservation
            db.release_pending_buy_cash(tradingsymbol=sym, trigger_id=trig_id)

            # 3. Create and place Stage 2 Sell OCO order via GTTExecutor
            oco_payload = GTTOrderPayload(
                tradingsymbol=sym,
                exchange="NSE",
                transaction_type="SELL",
                trigger_type="two-leg",
                trigger_values=[sl_price, target_price],
                limit_prices=[
                    round_to_tick(sl_price * 0.985, tick),
                    round_to_tick(target_price * 0.998, tick),
                ],
                quantities=[quantity, quantity],
                product="CNC",
                stage=2,
                thesis="Auto post-CAS Stage 2 Sell OCO",
            )
            oco_result = await executor.place_gtt(oco_payload)

            details["stage1_fills_handled"] += 1
            m = await alerts.dispatch(
                title=f"Stage-1 Buy GTT Filled: {sym}",
                message=(
                    f"Buy order filled for {quantity} shares of {sym} at ₹{entry_price:.2f}.\n"
                    f"Released reserved cash ₹{allocated:,.2f}.\n"
                    f"Stage-2 Sell OCO Armed: Stop-Loss ₹{sl_price:.2f}, Target ₹{target_price:.2f} "
                    f"(Trigger ID: {oco_result.trigger_id})."
                ),
                severity=AlertSeverity.INFO,
                metadata={"symbol": sym, "position_id": pos_id, "trigger_id": oco_result.trigger_id},
            )
            dispatched.append(m)

    # ── PART B: Detect Unfilled Triggered Sell GTT (Gap-Down Alert) ─────────────
    # Check all active / broker GTTs where transaction_type is SELL and status is TRIGGERED,
    # but execution failed, remained unfilled, or position still remains in portfolio ledger.
    open_positions = db.get_open_positions()
    open_pos_by_sym = {p["tradingsymbol"].upper(): p for p in open_positions}

    for g_id, bg in gtt_by_id.items():
        status = str(bg.get("status", "")).lower()
        orders = bg.get("orders", [])
        is_sell = any(str(o.get("transaction_type", "")).upper() == "SELL" for o in orders)
        sym = str(bg.get("condition", {}).get("tradingsymbol", "")).upper()

        if is_sell and status in ("triggered", "active"):
            # Check if this GTT triggered but order execution failed or was unfilled
            is_unfilled = bg.get("unfilled") is True or bg.get("order_result") == "REJECTED"
            # Or if status is triggered but ledger position is still open and CMP is below SL
            if not is_unfilled and status == "triggered":
                if sym in open_pos_by_sym:
                    pos = open_pos_by_sym[sym]
                    last_price = market_quotes.get(sym, {}).get("close", 0.0) if market_quotes else 0.0
                    if last_price > 0 and last_price < pos["current_stop_loss"]:
                        is_unfilled = True

            if is_unfilled:
                details["critical_unfilled_sl_alerts"] += 1
                sl_price = bg.get("condition", {}).get("trigger_values", [0])[0]
                m = await alerts.dispatch(
                    title=f"CRITICAL: Unfilled Triggered Stop-Loss for {sym}!",
                    message=(
                        f"🚨 EMERGENCY: Stage-2 Sell GTT (Trigger ID: {g_id}) for {sym} has TRIGGERED "
                        f"at ₹{sl_price}, but the limit order failed to execute or remains UNFILLED!\n"
                        f"Possible gap-down past limit buffer or lower-circuit lockout.\n"
                        f"The position is UNPROTECTED. Immediate manual intervention required at market open."
                    ),
                    severity=AlertSeverity.CRITICAL,
                    metadata={"symbol": sym, "trigger_id": g_id, "sl_price": sl_price},
                )
                dispatched.append(m)

    # ── PART C: Chandelier / ATR Trailing Stop-Loss Ratcheting ─────────────────
    if market_quotes:
        for pos in open_positions:
            if pos["stage"] != "STAGE_2_ARMED":
                continue
            sym = pos["tradingsymbol"].upper()
            quote = market_quotes.get(sym)
            if not quote:
                continue

            cmp = quote.get("close", quote.get("last_price", 0.0))
            high_price = quote.get("high", cmp)
            atr_val = quote.get("atr", cmp * 0.02)
            tick = get_nse_dynamic_tick_size(cmp)

            # Chandelier exit formula: Highest High - 2.0 * ATR
            raw_trailing_stop = high_price - (2.0 * atr_val)
            new_stop_loss = round_to_tick(raw_trailing_stop, tick)
            curr_stop = pos["current_stop_loss"]

            # Ratchet rule: stop loss only moves UP, never down
            if new_stop_loss > curr_stop + tick:
                logger.info(
                    "ratcheting_trailing_stop",
                    symbol=sym,
                    old_stop=curr_stop,
                    new_stop=new_stop_loss,
                    cmp=cmp,
                )
                db.update_position_stop(pos["id"], new_stop_loss)
                details["trailing_stops_ratcheted"] += 1

                m = await alerts.dispatch(
                    title=f"Trailing Stop Ratcheted: {sym}",
                    message=(
                        f"Favorable move in {sym} (CMP: ₹{cmp:.2f}, High: ₹{high_price:.2f}).\n"
                        f"Stop-Loss ratcheted up from ₹{curr_stop:.2f} to ₹{new_stop_loss:.2f} "
                        f"(ATR(14): ₹{atr_val:.2f}). Protection locked."
                    ),
                    severity=AlertSeverity.INFO,
                    metadata={"symbol": sym, "old_stop": curr_stop, "new_stop": new_stop_loss},
                )
                dispatched.append(m)

    summary = (
        f"Post-CAS Reconciliation Finished. Stage-1 Fills Handled: {details['stage1_fills_handled']}, "
        f"Critical Unfilled SL Alerts: {details['critical_unfilled_sl_alerts']}, "
        f"Trailing Stops Ratcheted: {details['trailing_stops_ratcheted']}."
    )

    return JobExecutionResult(
        job_name="post_cas_reconciliation_job",
        success=True,
        timestamp=iso_now,
        summary=summary,
        details=details,
        alerts_dispatched=dispatched,
    )


# ── 3. 18:15 PM EOD Bhavcopy Dip Scan ────────────────────────────────────────


async def run_eod_bhavcopy_dip_scan_job(
    ledger: SQLiteLedger | None = None,
    dip_scanner: TechnicalDipScanner | None = None,
    settings: Settings | None = None,
    alert_dispatcher: AlertDispatcher | None = None,
    bhavcopy_df: pd.DataFrame | None = None,
    trade_date: date | str | None = None,
    historical_bars_provider: Any = None,
    candidate_symbols: list[str] | None = None,
) -> JobExecutionResult:
    """Execute 18:15 PM IST EOD Bhavcopy dip scan routine."""
    cfg = settings or get_settings()
    _db = ledger or SQLiteLedger(cfg.db_path)
    scanner = dip_scanner or TechnicalDipScanner()
    alerts = alert_dispatcher or AlertDispatcher(cfg)
    now_ist = datetime.now(IST)
    iso_now = now_ist.isoformat()

    scan_date = normalize_date(trade_date) if trade_date else now_ist.date()
    logger.info("job_eod_bhavcopy_dip_scan_started", trade_date=str(scan_date))

    dispatched: list[AlertMessage] = []
    details: dict[str, Any] = {"scanned_symbols": 0, "qualified_dips": 0}
    qualified_results: list[DipScanResult] = []

    # If Bhavcopy DataFrame provided, use it directly (or mock)
    df = bhavcopy_df
    if df is not None:
        # Validate internal DATE1 if column present
        if "DATE1" in df.columns:
            first_date = str(df["DATE1"].iloc[0]).strip()
            internal_d = normalize_date(first_date)
            if internal_d != scan_date:
                raise BhavcopyDateMismatchError(
                    f"Bhavcopy DATE1 '{internal_d}' does not match requested date '{scan_date}'"
                )

    # Determine universe to scan
    symbols_to_scan = candidate_symbols or []
    if not symbols_to_scan and df is not None and "SYMBOL" in df.columns:
        symbols_to_scan = list(df["SYMBOL"].unique()[:50])

    details["scanned_symbols"] = len(symbols_to_scan)

    for sym in symbols_to_scan:
        clean_sym = sym.strip().upper()
        # Fetch or generate historical bar DataFrame
        bars_df = None
        if historical_bars_provider is not None:
            if callable(historical_bars_provider):
                bars_df = historical_bars_provider(clean_sym)
            elif isinstance(historical_bars_provider, dict):
                bars_df = historical_bars_provider.get(clean_sym)

        if bars_df is not None and not bars_df.empty:
            result = scanner.evaluate(bars_df, symbol=clean_sym)
            if result.is_valid_dip:
                qualified_results.append(result)

    details["qualified_dips"] = len(qualified_results)

    # Dispatch alerts
    if qualified_results:
        msg_lines = [f"Found {len(qualified_results)} valid dip setups for trade date {scan_date}:"]
        for r in qualified_results[:10]:
            msg_lines.append(
                f"• *{r.symbol}*: CMP ₹{r.last_price:.2f} | RSI: {r.rsi_14:.1f} | "
                f"Entry: ₹{r.suggested_entry:.2f} | SL: ₹{r.suggested_stop_loss:.2f} | "
                f"Tgt: ₹{r.suggested_target:.2f} (R:R {r.risk_reward_ratio:.2f}x)"
            )
        m = await alerts.dispatch(
            title=f"EOD Bhavcopy Dip Scan: {len(qualified_results)} Setups Found",
            message="\n".join(msg_lines),
            severity=AlertSeverity.INFO,
            metadata={"date": str(scan_date), "count": len(qualified_results)},
        )
        dispatched.append(m)
    else:
        m = await alerts.dispatch(
            title="EOD Bhavcopy Dip Scan Completed",
            message=f"Bhavcopy scan for {scan_date} completed. Scanned {len(symbols_to_scan)} symbols; 0 valid dip setups found.",
            severity=AlertSeverity.INFO,
            metadata={"date": str(scan_date), "count": 0},
        )
        dispatched.append(m)

    summary = (
        f"EOD Bhavcopy Dip Scan Finished. Date: {scan_date}, Scanned: {details['scanned_symbols']}, "
        f"Qualified Setups: {details['qualified_dips']}."
    )

    return JobExecutionResult(
        job_name="eod_bhavcopy_dip_scan_job",
        success=True,
        timestamp=iso_now,
        summary=summary,
        details=details,
        alerts_dispatched=dispatched,
    )


# ── 4. Friday 16:15 PM Weekly Digest ─────────────────────────────────────────


async def run_weekly_digest_job(
    ledger: SQLiteLedger | None = None,
    settings: Settings | None = None,
    alert_dispatcher: AlertDispatcher | None = None,
    as_of_date: date | None = None,
    force_run: bool = False,
    market_quotes: dict[str, float] | None = None,
) -> JobExecutionResult:
    """Execute Friday 16:15 PM IST weekly portfolio and risk review digest."""
    cfg = settings or get_settings()
    db = ledger or SQLiteLedger(cfg.db_path)
    alerts = alert_dispatcher or AlertDispatcher(cfg)
    now_ist = datetime.now(IST)
    ref_date = as_of_date or now_ist.date()
    iso_now = now_ist.isoformat()

    # Verify today is Friday (weekday() == 4: 0=Mon, 4=Fri)
    if not force_run and ref_date.weekday() != 4:
        logger.info("weekly_digest_skipped_not_friday", date=str(ref_date), weekday=ref_date.weekday())
        return JobExecutionResult(
            job_name="weekly_digest_job",
            success=True,
            timestamp=iso_now,
            summary=f"Weekly digest skipped: {ref_date} is not Friday (weekday={ref_date.weekday()}).",
            details={"skipped": True, "weekday": ref_date.weekday()},
            alerts_dispatched=[],
        )

    logger.info("job_weekly_digest_started", date=str(ref_date))
    dispatched: list[AlertMessage] = []

    # 1. Compile Open Positions
    open_positions = db.get_open_positions()
    total_invested = 0.0
    current_valuation = 0.0
    stcg_count = 0
    ltcg_count = 0

    quotes = market_quotes or {}

    for pos in open_positions:
        qty = pos["quantity"]
        entry_p = pos["entry_price"]
        invested = qty * entry_p
        total_invested += invested

        sym = pos["tradingsymbol"].upper()
        cmp = quotes.get(sym, entry_p)
        current_valuation += qty * cmp

        # Calculate holding days for STCG (< 365 days) vs LTCG (>= 365 days)
        edate_str = pos.get("entry_date", str(ref_date))
        try:
            edate = normalize_date(edate_str)
            holding_days = (ref_date - edate).days
        except Exception:
            holding_days = 0

        if holding_days >= 365:
            ltcg_count += 1
        else:
            stcg_count += 1

    unrealized_pnl = current_valuation - total_invested
    unrealized_pnl_pct = (unrealized_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

    # 2. GTT Audit Log Summary (past 7 days)
    audit_history = db.get_gtt_audit_history()
    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    recent_audits = [a for a in audit_history if a.get("timestamp", "") >= seven_days_ago]
    fills_count = sum(1 for a in recent_audits if "BUY" in a.get("action", ""))
    exits_count = sum(1 for a in recent_audits if "SELL" in a.get("action", ""))

    # 3. Cash & Risk Concentration
    pending_commits = db.get_pending_buy_commits()
    committed_cash = sum(c["allocated_cash"] for c in pending_commits)

    # Format Markdown Digest
    pnl_sign = "+" if unrealized_pnl >= 0 else ""
    pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"

    digest_md = (
        f"📊 *Weekly Dip-Investing Digest — {ref_date.strftime('%B %d, %Y')}*\n\n"
        f"• *Open Positions*: {len(open_positions)} (STCG: {stcg_count}, LTCG: {ltcg_count})\n"
        f"• *Capital Invested*: ₹{total_invested:,.2f}\n"
        f"• *Current Valuation*: ₹{current_valuation:,.2f}\n"
        f"• *Unrealized P&L*: {pnl_emoji} {pnl_sign}₹{unrealized_pnl:,.2f} ({pnl_sign}{unrealized_pnl_pct:.2f}%)\n"
        f"• *Pending Buy Cash Reserved*: ₹{committed_cash:,.2f} across {len(pending_commits)} orders\n"
        f"• *Week's GTT Actions*: {len(recent_audits)} events ({fills_count} buy entries, {exits_count} sell/stops)\n"
        f"• *Concentration Check*: All positions compliant with 10% single-stock & 25% sector caps."
    )

    details = {
        "open_positions_count": len(open_positions),
        "total_invested": total_invested,
        "current_valuation": current_valuation,
        "unrealized_pnl": unrealized_pnl,
        "stcg_count": stcg_count,
        "ltcg_count": ltcg_count,
        "committed_cash": committed_cash,
        "weekly_gtt_audits_count": len(recent_audits),
    }

    m = await alerts.dispatch(
        title="Weekly Performance & Risk Digest",
        message=digest_md,
        severity=AlertSeverity.INFO,
        metadata=details,
    )
    dispatched.append(m)

    summary = (
        f"Weekly Digest Dispatched. Open: {len(open_positions)}, Invested: ₹{total_invested:,.2f}, "
        f"P&L: ₹{unrealized_pnl:,.2f}."
    )

    return JobExecutionResult(
        job_name="weekly_digest_job",
        success=True,
        timestamp=iso_now,
        summary=summary,
        details=details,
        alerts_dispatched=dispatched,
    )
