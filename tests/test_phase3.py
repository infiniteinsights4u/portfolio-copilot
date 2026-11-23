"""Exhaustive pytest suite for Phase 3: Deterministic RiskGuard & Position Sizing Engine.

Covers:
1. Dynamic NSE tick-size table and tick rounding (NSE/CMTR/67133).
2. Zerodha Clause 1.6 minimum trigger distance (>=0.25% for >=₹50, >=₹0.09 for <₹50).
3. Circuit limit clamping and breach detection.
4. Worked Case A: ₹10L large-cap (68 sh, ₹0.10 tick, capital-cap binds at 9.98% / 0.63% risk).
5. Worked Case B: ₹10L mid-cap (158 sh, ₹0.05 tick, risk-cap binds at 9.53% / 1.00% risk).
6. 10% Single-stock concentration limit breach.
7. 25% Sector concentration limit breach.
8. Pending-GTT cash commitment reservation check (and SQLiteLedger integration).
9. 30-Second duplicate order suppression window.
10. BSE Corporate-action blackout (board meeting within 5 days, split, bonus, dividend > 2%).
11. Strict Stage-2 CNC holdings verification (quantity >= requested_sell_qty).
12. Zero quantity sized and invalid geometry rejection.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.risk.risk_guard import (
    GTTStage,
    OrderGeometry,
    RejectionReason,
    RiskGuard,
    RiskValidationRequest,
    RiskValidationResult,
    get_nse_dynamic_tick_size,
    is_tick_aligned,
    round_to_tick,
)
from src.storage.ledger import SQLiteLedger


# ── 1. Dynamic NSE Tick Sizing & Precision Rounding ───────────────────────────


class TestNSETickSizing:
    """Verify NSE dynamic tick bands and precision rounding per Circular NSE/CMTR/67133."""

    @pytest.mark.parametrize(
        "price, expected_tick",
        [
            (50.0, 0.01),
            (249.99, 0.01),
            (250.0, 0.05),
            (640.0, 0.05),
            (1000.0, 0.05),
            (1000.01, 0.10),
            (1550.0, 0.10),
            (5000.0, 0.10),
            (5000.01, 0.50),
            (8500.0, 0.50),
            (10000.0, 0.50),
            (10000.01, 1.00),
            (18500.0, 1.00),
            (20000.0, 1.00),
            (20000.01, 5.00),
            (35000.0, 5.00),
        ],
    )
    def test_nse_dynamic_tick_bands(self, price: float, expected_tick: float) -> None:
        """Verify dynamic tick resolution matches regulatory price brackets."""
        assert get_nse_dynamic_tick_size(price) == expected_tick

    def test_round_to_tick_precision(self) -> None:
        """Verify rounding to tick avoids floating-point binary artifacts."""
        # 0.10 tick
        assert round_to_tick(1467.30000001, 0.10) == 1467.30
        assert round_to_tick(1375.06, 0.10) == 1375.10
        assert round_to_tick(1616.76, 0.10) == 1616.80

        # 0.05 tick
        assert round_to_tick(603.00, 0.05) == 603.00
        assert round_to_tick(539.78, 0.05) == 539.80
        assert round_to_tick(728.54, 0.05) == 728.55

        # 0.01 tick
        assert round_to_tick(123.456, 0.01) == 123.46

        # 0.50 tick
        assert round_to_tick(6500.22, 0.50) == 6500.00
        assert round_to_tick(6500.26, 0.50) == 6500.50

        # 1.00 & 5.00 tick
        assert round_to_tick(12345.6, 1.00) == 12346.00
        assert round_to_tick(25112.3, 5.00) == 25110.00
        assert round_to_tick(25113.8, 5.00) == 25115.00

    def test_is_tick_aligned(self) -> None:
        """Verify alignment checker detects invalid off-tick prices."""
        assert is_tick_aligned(1375.10, 0.10) is True
        assert is_tick_aligned(1375.05, 0.10) is False  # Off-tick for ₹0.10 band!
        assert is_tick_aligned(603.05, 0.05) is True
        assert is_tick_aligned(603.02, 0.05) is False


# ── 2. Zerodha Clause 1.6 Proximity Validation ─────────────────────────────────


class TestClause16Proximity:
    """Verify minimum trigger distance rule from live CMP (Clause 1.6)."""

    def test_stocks_above_50_rupees(self) -> None:
        """Stocks >= ₹50 require >= 0.25% distance from CMP."""
        rg = RiskGuard()
        cmp = 1000.0  # 0.25% is ₹2.50

        # Exact CMP -> reject
        ok, err = rg.check_clause_1_6(1000.0, cmp)
        assert not ok
        assert "cannot be exactly equal" in (err or "")

        # Too close (< 0.25% = ₹2.50 distance)
        ok, err = rg.check_clause_1_6(1001.0, cmp)
        assert not ok
        assert "Clause 1.6 proximity violation" in (err or "")

        ok, err = rg.check_clause_1_6(998.0, cmp)
        assert not ok
        assert "Clause 1.6 proximity violation" in (err or "")

        # Sufficient distance (>= 0.25%)
        ok, err = rg.check_clause_1_6(1003.0, cmp)
        assert ok
        assert err is None

        ok, err = rg.check_clause_1_6(995.0, cmp)
        assert ok
        assert err is None

    def test_stocks_below_50_rupees(self) -> None:
        """Stocks < ₹50 require >= ₹0.09 distance from CMP."""
        rg = RiskGuard()
        cmp = 40.0

        # Exact CMP -> reject
        ok, err = rg.check_clause_1_6(40.0, cmp)
        assert not ok

        # Distance < ₹0.09 (e.g. ₹0.05)
        ok, err = rg.check_clause_1_6(40.05, cmp)
        assert not ok
        assert "sub-₹50 stock" in (err or "")

        # Distance >= ₹0.09
        ok, err = rg.check_clause_1_6(40.10, cmp)
        assert ok
        assert err is None

        ok, err = rg.check_clause_1_6(39.90, cmp)
        assert ok
        assert err is None


# ── 3. Worked Reference Cases (Case A & Case B) ────────────────────────────────


class TestWorkedReferenceCases:
    """Verify exact mathematical alignment with Image 10 and project specifications."""

    def test_case_a_large_cap_capital_cap_binds(self) -> None:
        """Case A: Lower-volatility large-cap (INFY) at ₹1,550, ATR ₹32, tick ₹0.10.

        Expected:
        - Buy Trigger: ₹1,460.00 -> Buy Limit (+0.50%): ₹1,467.30
        - SL Trigger (Entry - 2xATR): ₹1,396.00 -> SL Limit (-1.50%): ₹1,375.10
        - Risk per share: ₹92.20
        - Target Trigger: ₹1,620.00 -> Target Limit (-0.20%): ₹1,616.80
        - Q_risk: floor(10,000 / 92.20) = 108 shares
        - Q_capital: floor(100,000 / 1467.30) = 68 shares
        - Sized Qty: 68 shares (CAPITAL_CAP binds)
        - Capital Deployed: ₹99,776.40 (9.98%)
        - Worst-case Loss: ₹6,269.60 (0.63%)
        - Reward-to-Risk (worst limit): 1.62x
        """
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            tick_size=0.10,
        )

        result = rg.validate_and_size(request)

        assert result.is_valid is True
        assert len(result.rejection_reasons) == 0

        # Geometry checks
        geo = result.geometry
        assert geo is not None
        assert geo.buy_trigger == 1460.00
        assert geo.buy_limit == 1467.30
        assert geo.sl_trigger == 1396.00
        assert geo.sl_limit == 1375.10
        assert geo.target_trigger == 1620.00
        assert geo.target_limit == 1616.80
        assert geo.tick_size == 0.10

        # Sizing checks
        sz = result.sizing
        assert sz is not None
        assert sz.risk_cap_qty == 108
        assert sz.capital_cap_qty == 68
        assert sz.quantity == 68
        assert sz.binding_constraint == "CAPITAL_CAP"
        assert sz.worst_case_risk_per_share == 92.20
        assert sz.capital_deployed_inr == 99776.40
        assert sz.capital_deployed_pct == 0.0998  # 9.98%
        assert sz.worst_case_risk_inr == 6269.60
        assert sz.worst_case_risk_pct == 0.0063  # 0.63%
        assert sz.reward_risk_ratio_worst == 1.62

        # Proposal payload
        assert result.proposal_payload is not None
        assert result.proposal_payload["type"] == "single"
        assert result.proposal_payload["orders"][0]["quantity"] == 68
        assert result.proposal_payload["orders"][0]["price"] == 1467.30

    def test_case_b_mid_cap_risk_cap_binds(self) -> None:
        """Case B: Higher-volatility mid-cap at ₹640, ATR ₹26, tick ₹0.05.

        Expected:
        - Buy Trigger: ₹600.00 -> Buy Limit (+0.50%): ₹603.00
        - SL Trigger (Entry - 2xATR): ₹548.00 -> SL Limit (-1.50%): ₹539.80
        - Risk per share: ₹63.20
        - Target Trigger: ₹730.00 -> Target Limit (-0.20%): ₹728.55
        - Q_risk: floor(10,000 / 63.20) = 158 shares
        - Q_capital: floor(100,000 / 603.00) = 165 shares
        - Sized Qty: 158 shares (RISK_CAP binds)
        - Capital Deployed: ₹95,274.00 (9.53%)
        - Worst-case Loss: ₹9,985.60 (1.00% / ₹9,986)
        - Reward-to-Risk (worst limit): 1.99x
        """
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="TATAMOTORS",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=640.0,
            atr=26.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=500.0,
            upper_circuit_limit=800.0,
            buy_trigger=600.0,
            target_trigger=730.0,
            tick_size=0.05,
        )

        result = rg.validate_and_size(request)

        assert result.is_valid is True
        assert len(result.rejection_reasons) == 0

        # Geometry checks
        geo = result.geometry
        assert geo is not None
        assert geo.buy_trigger == 600.00
        assert geo.buy_limit == 603.00
        assert geo.sl_trigger == 548.00
        assert geo.sl_limit == 539.80
        assert geo.target_trigger == 730.00
        assert geo.target_limit == 728.55
        assert geo.tick_size == 0.05

        # Sizing checks
        sz = result.sizing
        assert sz is not None
        assert sz.risk_cap_qty == 158
        assert sz.capital_cap_qty == 165
        assert sz.quantity == 158
        assert sz.binding_constraint == "RISK_CAP"
        assert sz.worst_case_risk_per_share == 63.20
        assert sz.capital_deployed_inr == 95274.00
        assert sz.capital_deployed_pct == 0.0953  # 9.53%
        assert sz.worst_case_risk_inr == 9985.60
        assert sz.worst_case_risk_pct == 0.0100  # 1.00%
        assert sz.reward_risk_ratio_worst == 1.99


# ── 4. Circuit Limits Clamping & Breaches ───────────────────────────────────────


class TestCircuitLimits:
    """Verify circuit limit adherence and clamping."""

    def test_buy_limit_breaches_upper_circuit(self) -> None:
        """Reject if Buy limit exceeds upper circuit."""
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1465.0,  # Below buy_limit of 1467.30
            buy_trigger=1460.0,
            target_trigger=1620.0,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.CIRCUIT_LIMIT_BREACH in res.rejection_reasons

    def test_sl_limit_clamped_to_lower_circuit(self) -> None:
        """Verify SL limit is safely clamped to lower circuit if raw SL limit falls below."""
        rg = RiskGuard()
        # Raw SL trigger = 1460 - 64 = 1396. Raw SL limit = 1375.10.
        # Set lower circuit to 1380.00. SL limit must be clamped to 1380.00.
        geo = rg.compute_geometry(
            buy_trigger=1460.0,
            target_trigger=1620.0,
            atr=32.0,
            tick_size=0.10,
            lower_circuit=1380.0,
            upper_circuit=1800.0,
        )
        assert geo.sl_limit == 1380.00


# ── 5. Concentration & Cash Commitment Guardrails ──────────────────────────────


class TestConcentrationAndCashGuardrails:
    """Verify single-stock, sector, and pending-GTT cash ceilings."""

    def test_single_stock_10_percent_cap_exceeded(self) -> None:
        """Reject if existing holdings + new order exceeds 10% portfolio capital."""
        rg = RiskGuard()
        # Case A: new order is ₹99,776.40 (9.98%).
        # If user already holds 20 shares @ ₹1,550 (₹31,000), total exceeds ₹1,00,000.
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            existing_holding_qty=20,  # ₹31,000
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.CAPITAL_CAP_EXCEEDED in res.rejection_reasons
        assert any("Single-stock 10% cap exceeded" in s for s in res.rejection_details)

    def test_sector_25_percent_cap_exceeded(self) -> None:
        """Reject if existing sector exposure + new order exceeds 25% portfolio capital."""
        rg = RiskGuard()
        # Portfolio 10L. 25% cap is ₹2,50,000.
        # Existing IT sector value is ₹2,00,000.
        # New order is ₹99,776.40. Total = ₹2,99,776 > ₹2,50,000.
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            sector="Information Technology",
            existing_sector_value=200000.0,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.SECTOR_CAP_EXCEEDED in res.rejection_reasons
        assert any("Sector 25% cap exceeded" in s for s in res.rejection_details)

    def test_pending_buy_cash_commitment_exhaustion(self, tmp_path: Path) -> None:
        """Reject if active pending Buy GTT commitments + new order exceeds available cash."""
        # Setup SQLite ledger with pending buy commits
        db_file = tmp_path / "test_cash.db"
        ledger = SQLiteLedger(db_file)
        ledger.commit_pending_buy_cash("TCS", allocated_cash=850000.0)

        total_pending = ledger.get_total_pending_buy_cash()
        assert total_pending == 850000.0

        rg = RiskGuard()
        # Available cash = 900,000. Pending = 850,000. Remaining = 50,000.
        # Case A requires ~99,776.40 -> should reject!
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=900000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            total_pending_buy_cash=total_pending,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.INSUFFICIENT_CASH_PENDING_GTT in res.rejection_reasons
        assert any("Pending GTT cash commitment exceeded" in s for s in res.rejection_details)


# ── 6. 30-Second Duplicate Order Suppression ───────────────────────────────────


class TestDuplicateSuppression:
    """Verify suppression of identical orders placed within the 30-second window."""

    def test_duplicate_order_within_30s_is_suppressed(self) -> None:
        rg = RiskGuard(duplicate_window_sec=30)
        t0 = 1000.0

        req1 = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            custom_timestamp=t0,
        )
        res1 = rg.validate_and_size(req1)
        assert res1.is_valid is True

        # Second request 15 seconds later -> suppressed
        req2 = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            custom_timestamp=t0 + 15.0,
        )
        res2 = rg.validate_and_size(req2)
        assert res2.is_valid is False
        assert RejectionReason.DUPLICATE_ORDER_SUPPRESSED in res2.rejection_reasons

        # Third request 35 seconds later -> allowed
        req3 = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            custom_timestamp=t0 + 35.0,
        )
        res3 = rg.validate_and_size(req3)
        assert res3.is_valid is True


# ── 7. BSE Corporate Action Blackout ───────────────────────────────────────────


class TestCorporateActionBlackout:
    """Verify rejection during board meetings, splits, bonuses, or extraordinary dividends."""

    def test_board_meeting_within_5_days_blackout(self) -> None:
        rg = RiskGuard()
        today = date(2026, 11, 23)
        events = [
            {
                "category": "board_meeting",
                "title": "Board Meeting for Q2 Financial Results",
                "date": (today + timedelta(days=3)).isoformat(),
            }
        ]
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            corporate_events=events,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.CORPORATE_ACTION_BLACKOUT in res.rejection_reasons
        assert any("Board meeting scheduled" in s for s in res.rejection_details)

    def test_stock_split_blackout(self) -> None:
        rg = RiskGuard()
        today = date(2026, 11, 23)
        events = [
            {
                "category": "corporate_action",
                "title": "Sub-division / Stock Split of Equity Shares 1:5",
                "date": (today + timedelta(days=2)).isoformat(),
            }
        ]
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            corporate_events=events,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.CORPORATE_ACTION_BLACKOUT in res.rejection_reasons

    def test_extraordinary_dividend_above_2_percent_blackout(self) -> None:
        rg = RiskGuard()
        today = date(2026, 11, 23)
        # CMP is 1550. 2% is ₹31.00. Dividend of ₹40 exceeds 2%.
        events = [
            {
                "category": "dividend",
                "title": "Interim Dividend Declaration",
                "dividend_amount": 40.0,
                "date": (today + timedelta(days=4)).isoformat(),
            }
        ]
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            corporate_events=events,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.CORPORATE_ACTION_BLACKOUT in res.rejection_reasons

    def test_distant_event_is_not_blacked_out(self) -> None:
        rg = RiskGuard()
        today = date(2026, 11, 23)
        # Event 30 days away is outside 5-day blackout window
        events = [
            {
                "category": "board_meeting",
                "title": "Routine Board Meeting",
                "date": (today + timedelta(days=30)).isoformat(),
            }
        ]
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            corporate_events=events,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is True


# ── 8. Stage-2 CNC Holdings Check & Sell OCO Validation ────────────────────────


class TestStage2HoldingsCheck:
    """Verify strict Stage-2 CNC Sell OCO validation and holdings guardrail."""

    def test_insufficient_holdings_rejected(self) -> None:
        """Reject Stage-2 Sell OCO if live holdings < requested sell quantity."""
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_2_SELL_OCO,
            cmp=1500.0,
            atr=30.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1200.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            requested_sell_qty=50,
            existing_holding_qty=30,  # Only 30 shares in demat holdings!
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.STAGE_2_INSUFFICIENT_HOLDINGS in res.rejection_reasons
        assert any("Stage 2 CNC holding check failed" in s for s in res.rejection_details)

    def test_sufficient_holdings_creates_two_leg_oco_payload(self) -> None:
        """Approve Stage-2 Sell OCO when live holdings >= requested sell quantity."""
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_2_SELL_OCO,
            cmp=1500.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1200.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
            requested_sell_qty=68,
            existing_holding_qty=68,  # Exactly matches holdings
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is True
        assert res.proposal_payload is not None
        assert res.proposal_payload["type"] == "two-leg"
        assert res.proposal_payload["stage"] == GTTStage.STAGE_2_SELL_OCO.value

        orders = res.proposal_payload["orders"]
        assert len(orders) == 2
        assert orders[0]["transaction_type"] == "SELL"
        assert orders[0]["quantity"] == 68
        assert orders[0]["product"] == "CNC"
        assert orders[1]["transaction_type"] == "SELL"
        assert orders[1]["quantity"] == 68


# ── 9. Edge Cases & Sanity Rejections ──────────────────────────────────────────


class TestEdgeCases:
    """Verify defensive handling of zero sizing, zero ATR, and negative prices."""

    def test_zero_capital_sizes_zero_shares_and_rejects(self) -> None:
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=500.0,  # Too small to buy 1 share of ₹1467.30
            available_cash=500.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1620.0,
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.ZERO_QUANTITY_SIZED in res.rejection_reasons

    def test_invalid_target_below_buy_trigger_rejects(self) -> None:
        rg = RiskGuard()
        request = RiskValidationRequest(
            tradingsymbol="INFY",
            stage=GTTStage.STAGE_1_BUY_DIP,
            cmp=1550.0,
            atr=32.0,
            portfolio_capital=1000000.0,
            available_cash=1000000.0,
            lower_circuit_limit=1300.0,
            upper_circuit_limit=1800.0,
            buy_trigger=1460.0,
            target_trigger=1400.0,  # Below buy trigger!
        )
        res = rg.validate_and_size(request)
        assert res.is_valid is False
        assert RejectionReason.INVALID_PRICE_GEOMETRY in res.rejection_reasons
