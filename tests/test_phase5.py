"""Exhaustive pytest suite for Phase 5: Scheduled Positional Jobs & EOD Reconciliation.

Verifies:
1. Alert Dispatcher (multi-channel Telegram/Slack formatting, severity classifications,
   resilience against network errors, and dry-run collection).
2. 08:45 AM Pre-Market Sanity Check (Kite 6:00 AM session expiry warning, DDPI warning,
   corporate action blackout detection, and pending GTT reconciliation).
3. 16:05 PM Post-CAS Reconciliation:
   - Filled Stage-1 Buy GTT -> commits position, releases cash reservation, places Stage-2 Sell OCO.
   - Unfilled Triggered Sell GTT (Gap-Down / Circuit Breach) -> fires urgent CRITICAL alert.
   - Chandelier / ATR trailing stop ratcheting (ratchets up on favorable move, never down).
4. 18:15 PM EOD Bhavcopy Dip Scan (DATE1 validation, clean-room dip detection, setup geometry alerting).
5. Friday 16:15 PM Weekly Digest (Friday guard, STCG/LTCG tax categorisation, weekly GTT audit metrics).
6. JobScheduler execution engine & next run calculation.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.broker.gtt_executor import GTTExecutor
from src.config.settings import Settings
from src.data.bhavcopy import BhavcopyDateMismatchError
from src.scheduler.alerts import AlertDispatcher, AlertSeverity
from src.scheduler.jobs import (
    IST,
    run_eod_bhavcopy_dip_scan_job,
    run_post_cas_reconciliation_job,
    run_pre_market_job,
    run_weekly_digest_job,
)
from src.scheduler.scheduler import JobScheduler, compute_next_run
from src.storage.ledger import SQLiteLedger
from src.technicals.dip_scanner import TechnicalDipScanner


# ── 1. Alert Dispatcher Tests ────────────────────────────────────────────────


class TestAlertDispatcher:
    """Verify AlertDispatcher behavior, severity routing, and dry-run safety."""

    @pytest.mark.asyncio
    async def test_alert_dispatcher_in_memory_capture(self):
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)

        alert = await dispatcher.dispatch(
            title="System Heartbeat",
            message="All scheduled jobs operational.",
            severity=AlertSeverity.INFO,
            metadata={"source": "test"},
        )

        assert alert.title == "System Heartbeat"
        assert alert.severity == AlertSeverity.INFO
        assert len(dispatcher.dispatched_alerts) == 1
        assert "ℹ️ [INFO]" in alert.formatted_text()

    @pytest.mark.asyncio
    async def test_alert_severity_prefixes(self):
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)

        a_warn = await dispatcher.dispatch("Warning Title", "Details", AlertSeverity.WARNING)
        a_crit = await dispatcher.dispatch("Critical Title", "Details", AlertSeverity.CRITICAL)

        assert "⚠️ [WARNING]" in a_warn.formatted_text()
        assert "🚨 [CRITICAL ALERT]" in a_crit.formatted_text()

    @pytest.mark.asyncio
    async def test_alert_network_failure_resilience(self):
        """Simulate external Slack/Telegram HTTP failures and verify no crash."""
        settings = Settings(
            dry_run=False,
            slack_webhook_url="https://hooks.slack.com/services/FAKE/URL",
            telegram_bot_token="fake-token",
            telegram_chat_id="fake-chat-id",
        )
        dispatcher = AlertDispatcher(settings=settings, timeout_sec=0.1)

        with patch("httpx.AsyncClient.post", side_effect=Exception("Connection refused")):
            alert = await dispatcher.dispatch(
                title="Test Failure",
                message="Testing resilient fallback.",
                severity=AlertSeverity.WARNING,
            )
            assert alert.title == "Test Failure"
            assert len(dispatcher.dispatched_alerts) == 1

    def test_sync_dispatch_helper(self):
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        alert = dispatcher.dispatch_sync("Sync Alert", "Sync Message", AlertSeverity.INFO)
        assert alert.title == "Sync Alert"
        assert len(dispatcher.dispatched_alerts) == 1


# ── 2. 08:45 AM Pre-Market Sanity Check Tests ────────────────────────────────


class TestPreMarketJob:
    """Verify 08:45 AM IST pre-market routine."""

    @pytest.mark.asyncio
    async def test_session_expiry_triggers_warning(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True, zerodha_ddpi_enabled=True)
        dispatcher = AlertDispatcher(settings=settings)

        # Mock kite_client that fails on get_gtts (expired OAuth session)
        mock_kite = MagicMock()
        mock_kite.get_gtts = AsyncMock(side_effect=RuntimeError("Token expired at 06:00:00 IST"))

        res = await run_pre_market_job(
            ledger=db,
            kite_client=mock_kite,
            settings=settings,
            alert_dispatcher=dispatcher,
        )

        assert res.success is True
        assert res.details["session_healthy"] is False
        # Verify WARNING alert was dispatched
        warnings = [a for a in res.alerts_dispatched if a.severity == AlertSeverity.WARNING]
        assert any("Zerodha Kite Session Expired" in w.title for w in warnings)

    @pytest.mark.asyncio
    async def test_ddpi_disabled_triggers_tpin_warning(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True, zerodha_ddpi_enabled=False)
        dispatcher = AlertDispatcher(settings=settings)

        res = await run_pre_market_job(
            ledger=db,
            settings=settings,
            alert_dispatcher=dispatcher,
            skip_client_probe=True,
        )

        assert res.details["ddpi_enabled"] is False
        ddpi_alerts = [a for a in res.alerts_dispatched if "CDSL TPIN" in a.title]
        assert len(ddpi_alerts) == 1
        assert ddpi_alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_corporate_action_blackout_detection(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        # Add open position in TCS
        db.record_position("TCS", 3500.0, 10, 3400.0, 3750.0)

        settings = Settings(dry_run=True, zerodha_ddpi_enabled=True)
        dispatcher = AlertDispatcher(settings=settings)

        test_today = date(2026, 11, 24)
        events = [
            {
                "symbol": "TCS",
                "purpose": "Board Meeting to consider Financial Results",
                "date": "2026-11-26",
            }
        ]

        res = await run_pre_market_job(
            ledger=db,
            settings=settings,
            alert_dispatcher=dispatcher,
            today_override=test_today,
            corporate_events=events,
            skip_client_probe=True,
        )

        blackout_alerts = [a for a in res.alerts_dispatched if "Corporate Action Blackout" in a.title]
        assert len(blackout_alerts) == 1
        assert "TCS" in blackout_alerts[0].message


# ── 3. 16:05 PM Post-CAS Reconciliation Tests ────────────────────────────────


class TestPostCASReconciliationJob:
    """Verify Post-CAS reconciliation, Stage-2 OCO placement, gap-down SL alert, and trailing stops."""

    @pytest.mark.asyncio
    async def test_filled_stage1_buy_queues_stage2_sell_oco(self, tmp_path: Path):
        """Simulate filled Stage-1 Buy GTT -> verify cash release and Stage-2 Sell OCO placement."""
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        executor = GTTExecutor(ledger=db, settings=settings)

        # 1. Setup pending buy commit in ledger
        sym = "INFY"
        mock_trigger_id = 71234567
        db.commit_pending_buy_cash(tradingsymbol=sym, allocated_cash=50000.0, trigger_id=mock_trigger_id)
        assert len(db.get_pending_buy_commits()) == 1

        # 2. Broker GTT list showing the order has TRIGGERED / FILLED
        broker_gtts = [
            {
                "id": mock_trigger_id,
                "status": "triggered",
                "condition": {
                    "tradingsymbol": sym,
                    "trigger_values": [1500.0],
                },
                "orders": [
                    {
                        "transaction_type": "BUY",
                        "price": 1500.0,
                        "quantity": 33,
                    }
                ],
            }
        ]

        # 3. Run reconciliation job
        res = await run_post_cas_reconciliation_job(
            ledger=db,
            gtt_executor=executor,
            settings=settings,
            alert_dispatcher=dispatcher,
            broker_gtts=broker_gtts,
        )

        assert res.success is True
        assert res.details["stage1_fills_handled"] == 1

        # Assert cash was released from pending commits
        remaining_commits = db.get_pending_buy_commits()
        assert len(remaining_commits) == 0

        # Assert position was armed in positions_ledger
        positions = db.get_open_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos["tradingsymbol"] == "INFY"
        assert pos["stage"] == "STAGE_2_ARMED"
        assert pos["entry_price"] == 1500.0

        # Assert Stage-2 Sell OCO was recorded in GTT audit log
        gtt_audits = db.get_gtt_audit_history("INFY")
        oco_audits = [a for a in gtt_audits if a["stage"] == 2]
        assert len(oco_audits) >= 1
        assert oco_audits[0]["action"] == "SELL_OCO"

        # Assert Alert was dispatched
        fill_alerts = [a for a in res.alerts_dispatched if "Stage-1 Buy GTT Filled" in a.title]
        assert len(fill_alerts) == 1
        assert "INFY" in fill_alerts[0].message

    @pytest.mark.asyncio
    async def test_unfilled_triggered_sell_gtt_fires_critical_alert(self, tmp_path: Path):
        """Simulate gap-down past stop-loss / lower-circuit unfilled SL -> verify CRITICAL alert."""
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        executor = GTTExecutor(ledger=db, settings=settings)

        # Existing armed position for RELIANCE
        db.record_position(
            tradingsymbol="RELIANCE",
            entry_price=2500.0,
            quantity=20,
            current_stop_loss=2400.0,
            target_price=2750.0,
            stage="STAGE_2_ARMED",
        )

        # Broker reports Sell GTT triggered, but order failed/unfilled (e.g. stock gapped down past limit)
        sell_trig_id = 78901234
        broker_gtts = [
            {
                "id": sell_trig_id,
                "status": "triggered",
                "unfilled": True,
                "condition": {
                    "tradingsymbol": "RELIANCE",
                    "trigger_values": [2400.0],
                },
                "orders": [{"transaction_type": "SELL", "price": 2364.0, "quantity": 20}],
            }
        ]

        res = await run_post_cas_reconciliation_job(
            ledger=db,
            gtt_executor=executor,
            settings=settings,
            alert_dispatcher=dispatcher,
            broker_gtts=broker_gtts,
        )

        assert res.details["critical_unfilled_sl_alerts"] == 1
        crit_alerts = [a for a in res.alerts_dispatched if a.severity == AlertSeverity.CRITICAL]
        assert len(crit_alerts) == 1
        assert "CRITICAL: Unfilled Triggered Stop-Loss for RELIANCE" in crit_alerts[0].title
        assert "EMERGENCY" in crit_alerts[0].message

    @pytest.mark.asyncio
    async def test_trailing_stop_ratcheting_on_favorable_move(self, tmp_path: Path):
        """Simulate rally in price -> verify Chandelier/ATR trailing stop ratchets up."""
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        executor = GTTExecutor(ledger=db, settings=settings)

        # Initial position: entry ₹3,500, initial SL ₹3,400
        _pos_id = db.record_position(
            tradingsymbol="TCS",
            entry_price=3500.0,
            quantity=10,
            current_stop_loss=3400.0,
            target_price=3900.0,
            stage="STAGE_2_ARMED",
        )

        # Market rallies: High ₹3,850, Close ₹3,800, ATR ₹40
        # Chandelier stop: 3,850 - 2.0 * 40 = 3,770 (ratchets from 3,400 to 3,770)
        quotes = {
            "TCS": {
                "close": 3800.0,
                "high": 3850.0,
                "atr": 40.0,
            }
        }

        res = await run_post_cas_reconciliation_job(
            ledger=db,
            gtt_executor=executor,
            settings=settings,
            alert_dispatcher=dispatcher,
            broker_gtts=[],
            market_quotes=quotes,
        )

        assert res.details["trailing_stops_ratcheted"] == 1
        pos = db.get_open_positions()[0]
        assert pos["current_stop_loss"] == 3770.0

        ratchet_alerts = [a for a in res.alerts_dispatched if "Trailing Stop Ratcheted" in a.title]
        assert len(ratchet_alerts) == 1
        assert "₹3400.00 to ₹3770.00" in ratchet_alerts[0].message

    @pytest.mark.asyncio
    async def test_trailing_stop_never_ratchets_down(self, tmp_path: Path):
        """Simulate price drop -> verify trailing stop is never lowered."""
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        executor = GTTExecutor(ledger=db, settings=settings)

        # Position with current stop at ₹3,700
        db.record_position(
            tradingsymbol="TCS",
            entry_price=3500.0,
            quantity=10,
            current_stop_loss=3700.0,
            target_price=3900.0,
            stage="STAGE_2_ARMED",
        )

        # Market pulls back: High ₹3,650, Close ₹3,600, ATR ₹40
        # Chandelier stop: 3,650 - 80 = 3,570 (which is lower than 3,700)
        quotes = {
            "TCS": {
                "close": 3600.0,
                "high": 3650.0,
                "atr": 40.0,
            }
        }

        res = await run_post_cas_reconciliation_job(
            ledger=db,
            gtt_executor=executor,
            settings=settings,
            alert_dispatcher=dispatcher,
            broker_gtts=[],
            market_quotes=quotes,
        )

        assert res.details["trailing_stops_ratcheted"] == 0
        pos = db.get_open_positions()[0]
        assert pos["current_stop_loss"] == 3700.0  # Unchanged!


# ── 4. 18:15 PM EOD Bhavcopy Dip Scan Tests ──────────────────────────────────


class TestEODBhavcopyDipScanJob:
    """Verify EOD Bhavcopy ingestion and technical dip scan execution."""

    @pytest.mark.asyncio
    async def test_bhavcopy_date_mismatch_raises_error(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)

        # Bhavcopy CSV with internal DATE1 of 2026-11-20
        df = pd.DataFrame(
            {
                "SYMBOL": ["INFY"],
                "SERIES": ["EQ"],
                "DATE1": ["2026-11-20"],
                "CLOSE_PRICE": [1500.0],
            }
        )

        # Scan requested for 2026-11-24 -> must raise BhavcopyDateMismatchError
        with pytest.raises(BhavcopyDateMismatchError):
            await run_eod_bhavcopy_dip_scan_job(
                ledger=db,
                settings=settings,
                bhavcopy_df=df,
                trade_date=date(2026, 11, 24),
            )

    @pytest.mark.asyncio
    async def test_dip_scan_qualifies_candidate_and_dispatches_alert(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)

        # Create a synthetic 100-bar DataFrame for a stock in a clean dip
        n = 100
        # Trending up from 100 to 150, then gentle 2-bar pullback to 142
        base = [100.0 + (i * 0.5) for i in range(n - 2)]
        base.extend([142.0, 143.5])  # Pullback curl
        df_bars = pd.DataFrame(
            {
                "open": base,
                "high": [x + 2.0 for x in base],
                "low": [x - 1.5 for x in base],
                "close": base,
                "volume": [100000] * n,
                "DELIV_PER": [55.0] * n,
            }
        )

        bhav_df = pd.DataFrame(
            {
                "SYMBOL": ["QUALIFIED_STOCK"],
                "SERIES": ["EQ"],
                "DATE1": ["2026-11-24"],
                "CLOSE_PRICE": [143.5],
            }
        )

        scanner = TechnicalDipScanner()
        res = await run_eod_bhavcopy_dip_scan_job(
            ledger=db,
            dip_scanner=scanner,
            settings=settings,
            alert_dispatcher=dispatcher,
            bhavcopy_df=bhav_df,
            trade_date=date(2026, 11, 24),
            historical_bars_provider={"QUALIFIED_STOCK": df_bars},
            candidate_symbols=["QUALIFIED_STOCK"],
        )

        assert res.success is True
        assert res.details["scanned_symbols"] == 1
        assert len(res.alerts_dispatched) >= 1


# ── 5. Friday 16:15 PM Weekly Digest Tests ───────────────────────────────────


class TestWeeklyDigestJob:
    """Verify Friday weekly portfolio performance and risk digest."""

    @pytest.mark.asyncio
    async def test_weekly_digest_skipped_on_non_friday(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)

        # 2026-11-24 is a Tuesday (weekday == 1)
        res = await run_weekly_digest_job(
            ledger=db,
            settings=settings,
            alert_dispatcher=dispatcher,
            as_of_date=date(2026, 11, 24),
            force_run=False,
        )

        assert res.details["skipped"] is True
        assert len(res.alerts_dispatched) == 0

    @pytest.mark.asyncio
    async def test_weekly_digest_dispatches_on_friday_or_forced(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)

        # 1. Add STCG position (< 365 days)
        db.record_position(
            tradingsymbol="INFY",
            entry_price=1400.0,
            quantity=10,
            current_stop_loss=1350.0,
            target_price=1600.0,
            entry_date="2026-10-01",
        )
        # 2. Add LTCG position (>= 365 days)
        db.record_position(
            tradingsymbol="HDFCBANK",
            entry_price=1500.0,
            quantity=20,
            current_stop_loss=1450.0,
            target_price=1800.0,
            entry_date="2025-01-10",
        )

        # 3. Add pending buy commit
        db.commit_pending_buy_cash("TCS", 25000.0, trigger_id=999)

        # Friday date: 2026-11-27
        friday_date = date(2026, 11, 27)
        quotes = {"INFY": 1550.0, "HDFCBANK": 1650.0}

        res = await run_weekly_digest_job(
            ledger=db,
            settings=settings,
            alert_dispatcher=dispatcher,
            as_of_date=friday_date,
            market_quotes=quotes,
        )

        assert res.success is True
        assert res.details["open_positions_count"] == 2
        assert res.details["stcg_count"] == 1
        assert res.details["ltcg_count"] == 1
        assert res.details["committed_cash"] == 25000.0

        # Invested: (10*1400) + (20*1500) = 14000 + 30000 = 44000
        # Valuation: (10*1550) + (20*1650) = 15500 + 33000 = 48500
        # PnL: +4500
        assert res.details["total_invested"] == 44000.0
        assert res.details["current_valuation"] == 48500.0
        assert res.details["unrealized_pnl"] == 4500.0

        digest_alerts = [a for a in res.alerts_dispatched if "Weekly Performance & Risk Digest" in a.title]
        assert len(digest_alerts) == 1
        assert "LTCG: 1" in digest_alerts[0].message
        assert "₹4,500.00" in digest_alerts[0].message


# ── 6. Scheduler Engine Tests ────────────────────────────────────────────────


class TestJobScheduler:
    """Verify JobScheduler orchestration, trigger calculations, and error resilience."""

    def test_compute_next_run_calculation(self):
        ref_dt = datetime(2026, 11, 24, 8, 0, 0, tzinfo=IST)
        next_0845 = compute_next_run("08:45", current_dt=ref_dt)
        assert next_0845 == datetime(2026, 11, 24, 8, 45, 0, tzinfo=IST)

        # If current time is after 08:45, next run should be next day
        ref_dt_after = datetime(2026, 11, 24, 9, 0, 0, tzinfo=IST)
        next_day_0845 = compute_next_run("08:45", current_dt=ref_dt_after)
        assert next_day_0845 == datetime(2026, 11, 25, 8, 45, 0, tzinfo=IST)

        # Friday calculation (target_weekday=4)
        # 2026-11-24 is Tuesday (24 + 3 = 27 is Friday)
        next_fri = compute_next_run("16:15", target_weekday=4, current_dt=ref_dt)
        assert next_fri == datetime(2026, 11, 27, 16, 15, 0, tzinfo=IST)

    @pytest.mark.asyncio
    async def test_scheduler_executes_job_by_name(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True, zerodha_ddpi_enabled=True)
        scheduler = JobScheduler(settings=settings, ledger=db)

        res = await scheduler.execute_job("pre_market", skip_client_probe=True)
        assert res.job_name == "pre_market_job"
        assert res.success is True
        assert len(scheduler.get_history()) == 1

    @pytest.mark.asyncio
    async def test_scheduler_catches_unhandled_job_exceptions(self, tmp_path: Path):
        db = SQLiteLedger(tmp_path / "copilot.db")
        settings = Settings(dry_run=True)
        dispatcher = AlertDispatcher(settings=settings)
        scheduler = JobScheduler(settings=settings, ledger=db, alert_dispatcher=dispatcher)

        # Execute with an invalid job name
        res = await scheduler.execute_job("invalid_nonexistent_job")
        assert res.success is False
        assert "Unknown job name" in res.summary
        assert len(dispatcher.dispatched_alerts) == 1
        assert dispatcher.dispatched_alerts[0].severity == AlertSeverity.CRITICAL
