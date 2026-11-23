"""Deterministic pure-Python RiskGuard and Dual-Cap Position Sizer.

This module implements the complete 100% deterministic, zero-LLM risk validator
and position sizing engine for Zerodha CNC equity dip-investing as specified in:
- NSE Tick Size Revision Circular NSE/CMTR/67133
- Zerodha GTT Terms & Conditions Clause 1.6
- Dual-Cap Volatility Sizing: min(Q_risk, Q_capital)
- Market Protection (MPP) Limit Buffers (+0.50% Buy Limit, -1.50% SL Limit, -0.20% Target Limit)
- Live Circuit Limits Clamping
- 10% Single-Stock & 25% Sector Concentration Ceilings
- Pending-GTT Cash Commitment Reservation
- 30-Second Duplicate Order Suppression
- BSE Corporate Action Blackout (Board meetings within 5 days, splits/bonuses/rights/dividends > 2%)
- Stage-2 CNC Holdings Verification (quantity >= sell_qty)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping

import structlog

from src.config.settings import (
    DEFAULT_DUPLICATE_ORDER_WINDOW_SEC,
    DEFAULT_MAX_POSITION_SIZE_PCT,
    DEFAULT_MAX_RISK_PER_TRADE_PCT,
    DEFAULT_MAX_SECTOR_SIZE_PCT,
)

logger = structlog.get_logger(__name__)


class GTTStage(str, Enum):
    """GTT Order lifecycle stages."""

    STAGE_1_BUY_DIP = "STAGE_1_BUY_DIP"
    STAGE_2_SELL_OCO = "STAGE_2_SELL_OCO"


class RejectionReason(str, Enum):
    """Explicit enum-based rejection reasons emitted by RiskGuard."""

    INVALID_TICK_SIZE = "INVALID_TICK_SIZE"
    CLAUSE_1_6_PROXIMITY_VIOLATION = "CLAUSE_1_6_PROXIMITY_VIOLATION"
    CIRCUIT_LIMIT_BREACH = "CIRCUIT_LIMIT_BREACH"
    CAPITAL_CAP_EXCEEDED = "CAPITAL_CAP_EXCEEDED"
    SECTOR_CAP_EXCEEDED = "SECTOR_CAP_EXCEEDED"
    INSUFFICIENT_CASH_PENDING_GTT = "INSUFFICIENT_CASH_PENDING_GTT"
    DUPLICATE_ORDER_SUPPRESSED = "DUPLICATE_ORDER_SUPPRESSED"
    CORPORATE_ACTION_BLACKOUT = "CORPORATE_ACTION_BLACKOUT"
    STAGE_2_INSUFFICIENT_HOLDINGS = "STAGE_2_INSUFFICIENT_HOLDINGS"
    INVALID_PRICE_GEOMETRY = "INVALID_PRICE_GEOMETRY"
    ZERO_QUANTITY_SIZED = "ZERO_QUANTITY_SIZED"


# ── Dynamic NSE Tick Size (NSE Circular NSE/CMTR/67133) ────────────────────────


def get_nse_dynamic_tick_size(price: float) -> float:
    """Return the statutory NSE cash segment tick size based on price band.

    Bands:
    - Below ₹250.00: ₹0.01
    - ₹250.00 to ₹1,000.00: ₹0.05
    - > ₹1,000.00 to ₹5,000.00: ₹0.10
    - > ₹5,000.00 to ₹10,000.00: ₹0.50
    - > ₹10,000.00 to ₹20,000.00: ₹1.00
    - Above ₹20,000.00: ₹5.00
    """
    if price < 250.0:
        return 0.01
    if price <= 1000.0:
        return 0.05
    if price <= 5000.0:
        return 0.10
    if price <= 10000.0:
        return 0.50
    if price <= 20000.0:
        return 1.00
    return 5.00


def round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest tick increment using Decimal precision."""
    if tick_size <= 0.0:
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    d_price = Decimal(str(price))
    d_tick = Decimal(str(tick_size))
    units = (d_price / d_tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    rounded = units * d_tick

    # Determine decimal places of tick for clean float representation
    tick_str = str(tick_size).rstrip("0")
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
    return round(float(rounded), decimals)


def is_tick_aligned(price: float, tick_size: float, tolerance: float = 1e-5) -> bool:
    """Verify whether a price conforms to the required tick increment."""
    expected = round_to_tick(price, tick_size)
    return abs(price - expected) < tolerance


# ── Data Containers ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OrderGeometry:
    """Computed tick-aligned trigger and limit prices."""

    buy_trigger: float
    buy_limit: float
    sl_trigger: float
    sl_limit: float
    target_trigger: float
    target_limit: float
    tick_size: float
    buy_limit_buffer_pct: float = 0.0050   # +0.50%
    sl_limit_buffer_pct: float = 0.0150    # -1.50%
    target_limit_buffer_pct: float = 0.0020  # -0.20%


@dataclass(frozen=True)
class SizingResult:
    """Computed dual-cap position size and risk metrics."""

    quantity: int
    risk_cap_qty: int
    capital_cap_qty: int
    binding_constraint: str  # "CAPITAL_CAP" | "RISK_CAP"
    worst_case_risk_per_share: float
    worst_case_risk_inr: float
    worst_case_risk_pct: float
    capital_deployed_inr: float
    capital_deployed_pct: float
    reward_risk_ratio_worst: float
    reward_risk_ratio_trigger: float


@dataclass
class RiskValidationRequest:
    """Input payload submitted to RiskGuard for validation and sizing."""

    tradingsymbol: str
    stage: GTTStage
    cmp: float  # Current Market Price (LTP)
    atr: float  # Wilder 14-day ATR
    portfolio_capital: float
    available_cash: float
    lower_circuit_limit: float
    upper_circuit_limit: float
    isin: str | None = None
    sector: str | None = None
    buy_trigger: float | None = None
    target_trigger: float | None = None
    requested_sell_qty: int | None = None
    existing_holding_qty: int = 0
    existing_sector_value: float = 0.0
    pending_stock_commits: float = 0.0
    pending_sector_commits: float = 0.0
    total_pending_buy_cash: float = 0.0
    tick_size: float | None = None
    corporate_events: list[dict[str, Any]] = field(default_factory=list)
    custom_timestamp: float | None = None


@dataclass
class RiskValidationResult:
    """Output validation verdict emitted by RiskGuard."""

    is_valid: bool
    rejection_reasons: list[RejectionReason]
    rejection_details: list[str]
    geometry: OrderGeometry | None = None
    sizing: SizingResult | None = None
    proposal_payload: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ── RiskGuard Engine ───────────────────────────────────────────────────────────


class RiskGuard:
    """Deterministic, zero-LLM risk guardrail and dual-cap position sizing validator."""

    def __init__(
        self,
        max_risk_pct: float = DEFAULT_MAX_RISK_PER_TRADE_PCT,
        max_position_pct: float = DEFAULT_MAX_POSITION_SIZE_PCT,
        max_sector_pct: float = DEFAULT_MAX_SECTOR_SIZE_PCT,
        duplicate_window_sec: int = DEFAULT_DUPLICATE_ORDER_WINDOW_SEC,
    ) -> None:
        self.max_risk_pct = max_risk_pct
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.duplicate_window_sec = duplicate_window_sec
        self._recent_proposals: dict[tuple[str, str], float] = {}

    def reset_duplicate_history(self) -> None:
        """Clear recent proposal history (useful for unit testing)."""
        self._recent_proposals.clear()

    # ── Geometry and Sizing Math ───────────────────────────────────────────────

    def compute_geometry(
        self,
        buy_trigger: float,
        target_trigger: float,
        atr: float,
        tick_size: float,
        lower_circuit: float,
        upper_circuit: float,
    ) -> OrderGeometry:
        """Compute tick-aligned trigger and limit prices with circuit clamping.

        Geometry Rules:
        - Buy Limit: +0.50% above Buy Trigger, rounded to tick.
        - SL Trigger: Buy Trigger - 2 * ATR, rounded to tick.
        - SL Limit: -1.50% below SL Trigger, rounded to tick, clamped >= lower_circuit.
        - Target Limit: -0.20% below Target Trigger, rounded to tick, clamped <= upper_circuit.
        """
        # 1. Buy limit (+0.50% buffer)
        raw_buy_limit = buy_trigger * 1.0050
        buy_limit = round_to_tick(raw_buy_limit, tick_size)

        # 2. SL trigger (2 * ATR dip stop)
        raw_sl_trigger = buy_trigger - (2.0 * atr)
        sl_trigger = round_to_tick(raw_sl_trigger, tick_size)

        # 3. SL limit (-1.50% buffer, clamped to lower circuit)
        raw_sl_limit = sl_trigger * 0.9850
        sl_limit = round_to_tick(raw_sl_limit, tick_size)
        if sl_limit < lower_circuit:
            sl_limit = round_to_tick(lower_circuit, tick_size)

        # 4. Target limit (-0.20% buffer, clamped to upper circuit)
        raw_target_limit = target_trigger * 0.9980
        target_limit = round_to_tick(raw_target_limit, tick_size)
        if target_limit > upper_circuit:
            target_limit = round_to_tick(upper_circuit, tick_size)

        return OrderGeometry(
            buy_trigger=round_to_tick(buy_trigger, tick_size),
            buy_limit=buy_limit,
            sl_trigger=sl_trigger,
            sl_limit=sl_limit,
            target_trigger=round_to_tick(target_trigger, tick_size),
            target_limit=target_limit,
            tick_size=tick_size,
        )

    def calculate_dual_cap_size(
        self,
        portfolio_capital: float,
        geometry: OrderGeometry,
        max_risk_pct: float | None = None,
        max_position_pct: float | None = None,
    ) -> SizingResult:
        """Calculate share count Q = min(Q_risk, Q_capital) against worst-case loss."""
        r_pct = max_risk_pct if max_risk_pct is not None else self.max_risk_pct
        c_pct = max_position_pct if max_position_pct is not None else self.max_position_pct

        max_risk_inr = portfolio_capital * r_pct
        max_capital_inr = portfolio_capital * c_pct

        worst_case_risk_per_share = round(geometry.buy_limit - geometry.sl_limit, 4)
        if worst_case_risk_per_share <= 0:
            raise ValueError(
                f"Invalid risk per share: Buy limit ({geometry.buy_limit}) <= SL limit ({geometry.sl_limit})"
            )

        q_risk = math.floor(max_risk_inr / worst_case_risk_per_share)
        q_capital = math.floor(max_capital_inr / geometry.buy_limit)

        quantity = min(q_risk, q_capital)
        binding_constraint = "CAPITAL_CAP" if q_capital <= q_risk else "RISK_CAP"

        worst_case_risk_inr = round(quantity * worst_case_risk_per_share, 2)
        worst_case_risk_pct = round(worst_case_risk_inr / portfolio_capital, 4) if portfolio_capital > 0 else 0.0

        capital_deployed_inr = round(quantity * geometry.buy_limit, 2)
        capital_deployed_pct = round(capital_deployed_inr / portfolio_capital, 4) if portfolio_capital > 0 else 0.0

        worst_reward_per_share = geometry.target_limit - geometry.buy_limit
        rr_worst = round(worst_reward_per_share / worst_case_risk_per_share, 2) if worst_case_risk_per_share > 0 else 0.0

        trigger_risk = geometry.buy_trigger - geometry.sl_trigger
        trigger_reward = geometry.target_trigger - geometry.buy_trigger
        rr_trigger = round(trigger_reward / trigger_risk, 2) if trigger_risk > 0 else 0.0

        return SizingResult(
            quantity=quantity,
            risk_cap_qty=q_risk,
            capital_cap_qty=q_capital,
            binding_constraint=binding_constraint,
            worst_case_risk_per_share=worst_case_risk_per_share,
            worst_case_risk_inr=worst_case_risk_inr,
            worst_case_risk_pct=worst_case_risk_pct,
            capital_deployed_inr=capital_deployed_inr,
            capital_deployed_pct=capital_deployed_pct,
            reward_risk_ratio_worst=rr_worst,
            reward_risk_ratio_trigger=rr_trigger,
        )

    # ── Guardrail Rules ────────────────────────────────────────────────────────

    def check_clause_1_6(self, trigger_price: float, cmp: float) -> tuple[bool, str | None]:
        """Verify Zerodha GTT Clause 1.6 minimum proximity distance.

        Rules:
        - If CMP >= ₹50.00: |trigger - cmp| / cmp >= 0.0025 (0.25%)
        - If CMP < ₹50.00: |trigger - cmp| >= ₹0.09
        - Trigger price cannot equal CMP.
        """
        diff = abs(trigger_price - cmp)
        if diff == 0.0:
            return False, f"Trigger price ({trigger_price}) cannot be exactly equal to CMP ({cmp})"

        if cmp >= 50.0:
            required = round(cmp * 0.0025, 4)
            if diff < (required - 1e-5):
                return (
                    False,
                    f"Clause 1.6 proximity violation: |trigger ({trigger_price}) - cmp ({cmp})| = {diff:.2f} "
                    f"is less than required 0.25% distance ({required:.2f})",
                )
        else:
            required = 0.09
            if diff < (required - 1e-5):
                return (
                    False,
                    f"Clause 1.6 proximity violation: |trigger ({trigger_price}) - cmp ({cmp})| = {diff:.2f} "
                    f"is less than required ₹0.09 distance ({required:.2f}) for sub-₹50 stock",
                )

        return True, None

    def check_circuit_limits(
        self,
        prices: Mapping[str, float],
        lower_circuit: float,
        upper_circuit: float,
    ) -> tuple[bool, list[str]]:
        """Verify that proposed prices stay strictly within circuit bands."""
        errors: list[str] = []
        for name, price in prices.items():
            if price < lower_circuit:
                errors.append(f"{name} ({price}) is below lower circuit limit ({lower_circuit})")
            if price > upper_circuit:
                errors.append(f"{name} ({price}) is above upper circuit limit ({upper_circuit})")
        return len(errors) == 0, errors

    def check_duplicate_suppression(
        self,
        tradingsymbol: str,
        stage: GTTStage,
        now_ts: float | None = None,
    ) -> tuple[bool, str | None]:
        """Suppress identical GTT orders placed within the 30-second window."""
        ts = now_ts if now_ts is not None else time.time()
        key = (tradingsymbol.upper(), stage.value)
        last_ts = self._recent_proposals.get(key)

        if last_ts is not None:
            elapsed = ts - last_ts
            if elapsed < self.duplicate_window_sec:
                return (
                    False,
                    f"Duplicate order suppressed for {tradingsymbol} ({stage.value}): "
                    f"{elapsed:.1f}s elapsed < {self.duplicate_window_sec}s window",
                )

        return True, None

    def record_proposal(
        self,
        tradingsymbol: str,
        stage: GTTStage,
        now_ts: float | None = None,
    ) -> None:
        """Record timestamp of a successfully validated proposal."""
        ts = now_ts if now_ts is not None else time.time()
        self._recent_proposals[(tradingsymbol.upper(), stage.value)] = ts

    def check_corporate_actions_blackout(
        self,
        events: list[dict[str, Any]],
        cmp: float,
        blackout_days: int = 5,
        current_date: date | None = None,
    ) -> tuple[bool, list[str]]:
        """Check for upcoming high-impact BSE corporate events causing GTT cancellation or shock.

        Blackout Triggers:
        - Board meetings scheduled within next 5 trading days (earnings/results).
        - Stock splits, bonus issues, or rights issues.
        - Extraordinary dividends exceeding 2.0% of market price.
        """
        today = current_date or date.today()
        cutoff_date = today + timedelta(days=blackout_days)
        reasons: list[str] = []

        for ev in events:
            # Parse event date
            raw_date = ev.get("date")
            ev_date: date | None = None
            if isinstance(raw_date, str):
                try:
                    ev_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
                except ValueError:
                    try:
                        ev_date = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
                    except ValueError:
                        pass
            elif isinstance(raw_date, (datetime, date)):
                ev_date = raw_date if isinstance(raw_date, date) else raw_date.date()

            # Ignore events outside the forward blackout window if date is present
            if ev_date and (ev_date < today or ev_date > cutoff_date):
                continue

            category = str(ev.get("category", "")).lower()
            title = str(ev.get("title", "")).lower()
            desc = str(ev.get("description", "")).lower()
            text = f"{category} {title} {desc}"

            # 1. Earnings / Board Meeting check
            if "board meeting" in text or "board_meeting" in category:
                if any(w in text for w in ("result", "financial", "earnings", "dividend", "quarter")):
                    reasons.append(
                        f"Corporate Action Blackout: Board meeting scheduled on {ev_date or 'upcoming'} ({title})"
                    )

            # 2. Stock Splits
            if "split" in text or "sub-division" in text or "subdivision" in text:
                reasons.append(
                    f"Corporate Action Blackout: Stock split / sub-division announcement ({title})"
                )

            # 3. Bonus Issue
            if "bonus" in text:
                reasons.append(f"Corporate Action Blackout: Bonus issue announcement ({title})")

            # 4. Rights Issue
            if "rights" in text and "issue" in text:
                reasons.append(f"Corporate Action Blackout: Rights issue announcement ({title})")

            # 5. Extraordinary Dividend > 2% of CMP
            if "dividend" in text:
                div_amount = ev.get("dividend_amount")
                if div_amount is not None:
                    try:
                        div_amt = float(div_amount)
                        if cmp > 0 and (div_amt / cmp) > 0.02:
                            reasons.append(
                                f"Corporate Action Blackout: Dividend of ₹{div_amt} exceeds 2% of CMP ₹{cmp}"
                            )
                    except (ValueError, TypeError):
                        pass

        return len(reasons) == 0, reasons

    # ── Main Validation Entry Point ────────────────────────────────────────────

    def validate_and_size(self, request: RiskValidationRequest) -> RiskValidationResult:
        """Execute complete deterministic validation and position sizing pipeline."""
        rejection_reasons: list[RejectionReason] = []
        rejection_details: list[str] = []
        diagnostics: dict[str, Any] = {}

        # 1. Resolve Tick Size
        tick = request.tick_size or get_nse_dynamic_tick_size(request.cmp)
        diagnostics["tick_size"] = tick

        # 2. 30-Second Duplicate Suppression Check
        dup_ok, dup_err = self.check_duplicate_suppression(
            request.tradingsymbol, request.stage, request.custom_timestamp
        )
        if not dup_ok and dup_err:
            rejection_reasons.append(RejectionReason.DUPLICATE_ORDER_SUPPRESSED)
            rejection_details.append(dup_err)

        # 3. BSE Corporate Action Blackout Check
        if request.corporate_events:
            corp_ok, corp_errs = self.check_corporate_actions_blackout(
                request.corporate_events, request.cmp
            )
            if not corp_ok:
                rejection_reasons.append(RejectionReason.CORPORATE_ACTION_BLACKOUT)
                rejection_details.extend(corp_errs)

        # 4. Stage-specific logic
        if request.stage == GTTStage.STAGE_2_SELL_OCO:
            return self._validate_stage_2_sell(request, tick, rejection_reasons, rejection_details, diagnostics)
        else:
            return self._validate_stage_1_buy(request, tick, rejection_reasons, rejection_details, diagnostics)

    def _validate_stage_1_buy(
        self,
        request: RiskValidationRequest,
        tick: float,
        rejection_reasons: list[RejectionReason],
        rejection_details: list[str],
        diagnostics: dict[str, Any],
    ) -> RiskValidationResult:
        """Validate Stage 1 Single-Leg Buy GTT proposal and compute dual-cap size."""
        if request.buy_trigger is None:
            rejection_reasons.append(RejectionReason.INVALID_PRICE_GEOMETRY)
            rejection_details.append("Stage 1 Buy proposal requires buy_trigger")
            return RiskValidationResult(
                is_valid=False,
                rejection_reasons=rejection_reasons,
                rejection_details=rejection_details,
                diagnostics=diagnostics,
            )

        # Target trigger fallback if not specified: default 2.5R trigger target
        target_trigger = request.target_trigger
        if target_trigger is None:
            target_trigger = request.buy_trigger + (2.5 * 2.0 * request.atr)

        # Basic geometry sanity
        if request.buy_trigger <= 0:
            rejection_reasons.append(RejectionReason.INVALID_PRICE_GEOMETRY)
            rejection_details.append(f"buy_trigger ({request.buy_trigger}) must be > 0")

        if target_trigger <= request.buy_trigger:
            rejection_reasons.append(RejectionReason.INVALID_PRICE_GEOMETRY)
            rejection_details.append(
                f"target_trigger ({target_trigger}) must be strictly > buy_trigger ({request.buy_trigger})"
            )

        if request.atr <= 0:
            rejection_reasons.append(RejectionReason.INVALID_PRICE_GEOMETRY)
            rejection_details.append(f"ATR ({request.atr}) must be > 0")

        # Clause 1.6 Proximity Check on Buy Trigger vs CMP
        c16_ok, c16_err = self.check_clause_1_6(request.buy_trigger, request.cmp)
        if not c16_ok and c16_err:
            rejection_reasons.append(RejectionReason.CLAUSE_1_6_PROXIMITY_VIOLATION)
            rejection_details.append(c16_err)

        # Compute full Order Geometry
        geometry = self.compute_geometry(
            buy_trigger=request.buy_trigger,
            target_trigger=target_trigger,
            atr=request.atr,
            tick_size=tick,
            lower_circuit=request.lower_circuit_limit,
            upper_circuit=request.upper_circuit_limit,
        )

        # Verify Tick Alignment
        for name, val in [
            ("buy_trigger", geometry.buy_trigger),
            ("buy_limit", geometry.buy_limit),
            ("sl_trigger", geometry.sl_trigger),
            ("sl_limit", geometry.sl_limit),
            ("target_trigger", geometry.target_trigger),
            ("target_limit", geometry.target_limit),
        ]:
            if not is_tick_aligned(val, tick):
                rejection_reasons.append(RejectionReason.INVALID_TICK_SIZE)
                rejection_details.append(f"{name} ({val}) is not aligned to tick size {tick}")

        # Circuit Limits Check
        circ_ok, circ_errs = self.check_circuit_limits(
            {
                "buy_trigger": geometry.buy_trigger,
                "buy_limit": geometry.buy_limit,
                "sl_trigger": geometry.sl_trigger,
            },
            request.lower_circuit_limit,
            request.upper_circuit_limit,
        )
        if not circ_ok:
            rejection_reasons.append(RejectionReason.CIRCUIT_LIMIT_BREACH)
            rejection_details.extend(circ_errs)

        # Sizing Calculation
        sizing: SizingResult | None = None
        try:
            sizing = self.calculate_dual_cap_size(
                portfolio_capital=request.portfolio_capital,
                geometry=geometry,
            )
        except Exception as e:
            rejection_reasons.append(RejectionReason.INVALID_PRICE_GEOMETRY)
            rejection_details.append(f"Sizing calculation failed: {e}")

        if sizing is not None:
            diagnostics["sizing"] = sizing
            if sizing.quantity <= 0:
                rejection_reasons.append(RejectionReason.ZERO_QUANTITY_SIZED)
                rejection_details.append(
                    f"Sized quantity is 0 (risk_cap_qty={sizing.risk_cap_qty}, capital_cap_qty={sizing.capital_cap_qty})"
                )

            # Concentration Caps Verification
            new_order_val = sizing.capital_deployed_inr

            # 1. Single-stock 10% concentration cap
            total_stock_exposure = (
                (request.existing_holding_qty * request.cmp)
                + request.pending_stock_commits
                + new_order_val
            )
            max_stock_val = request.portfolio_capital * self.max_position_pct
            if total_stock_exposure > (max_stock_val + 1.0):  # +1.0 rupee rounding tolerance
                rejection_reasons.append(RejectionReason.CAPITAL_CAP_EXCEEDED)
                rejection_details.append(
                    f"Single-stock 10% cap exceeded: total exposure ₹{total_stock_exposure:.2f} "
                    f"> limit ₹{max_stock_val:.2f}"
                )

            # 2. Sector 25% concentration cap
            total_sector_exposure = (
                request.existing_sector_value
                + request.pending_sector_commits
                + new_order_val
            )
            max_sector_val = request.portfolio_capital * self.max_sector_pct
            if total_sector_exposure > (max_sector_val + 1.0):
                rejection_reasons.append(RejectionReason.SECTOR_CAP_EXCEEDED)
                rejection_details.append(
                    f"Sector 25% cap exceeded: total exposure ₹{total_sector_exposure:.2f} "
                    f"> limit ₹{max_sector_val:.2f}"
                )

            # 3. Pending-GTT Cash Reservation Check
            total_cash_needed = request.total_pending_buy_cash + new_order_val
            if total_cash_needed > request.available_cash:
                rejection_reasons.append(RejectionReason.INSUFFICIENT_CASH_PENDING_GTT)
                rejection_details.append(
                    f"Pending GTT cash commitment exceeded: total committed ₹{total_cash_needed:.2f} "
                    f"> available cash ₹{request.available_cash:.2f}"
                )

        is_valid = len(rejection_reasons) == 0

        proposal_payload: dict[str, Any] | None = None
        if is_valid and sizing is not None:
            self.record_proposal(request.tradingsymbol, request.stage, request.custom_timestamp)
            proposal_payload = {
                "type": "single",
                "condition": {
                    "exchange": "NSE",
                    "tradingsymbol": request.tradingsymbol.upper(),
                    "trigger_values": [geometry.buy_trigger],
                    "last_price": request.cmp,
                },
                "orders": [
                    {
                        "exchange": "NSE",
                        "tradingsymbol": request.tradingsymbol.upper(),
                        "transaction_type": "BUY",
                        "quantity": sizing.quantity,
                        "order_type": "LIMIT",
                        "product": "CNC",
                        "price": geometry.buy_limit,
                    }
                ],
                "stage": request.stage.value,
                "worst_case_risk_inr": sizing.worst_case_risk_inr,
                "capital_deployed_inr": sizing.capital_deployed_inr,
            }

        return RiskValidationResult(
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            rejection_details=rejection_details,
            geometry=geometry,
            sizing=sizing,
            proposal_payload=proposal_payload,
            diagnostics=diagnostics,
        )

    def _validate_stage_2_sell(
        self,
        request: RiskValidationRequest,
        tick: float,
        rejection_reasons: list[RejectionReason],
        rejection_details: list[str],
        diagnostics: dict[str, Any],
    ) -> RiskValidationResult:
        """Validate Stage 2 Two-Leg Sell OCO Stop-Loss + Target proposal."""
        sell_qty = request.requested_sell_qty or 0
        if sell_qty <= 0:
            rejection_reasons.append(RejectionReason.ZERO_QUANTITY_SIZED)
            rejection_details.append(f"requested_sell_qty ({sell_qty}) must be > 0 for Stage 2 Sell OCO")

        # Strict Holdings Verification: holdings must exist and be >= sell_qty
        if request.existing_holding_qty < sell_qty:
            rejection_reasons.append(RejectionReason.STAGE_2_INSUFFICIENT_HOLDINGS)
            rejection_details.append(
                f"Stage 2 CNC holding check failed: live holding ({request.existing_holding_qty}) "
                f"< requested sell quantity ({sell_qty})"
            )

        # SL and Target trigger logic
        buy_entry = request.buy_trigger or request.cmp
        target_trigger = request.target_trigger or (buy_entry + (2.5 * 2.0 * request.atr))

        geometry = self.compute_geometry(
            buy_trigger=buy_entry,
            target_trigger=target_trigger,
            atr=request.atr,
            tick_size=tick,
            lower_circuit=request.lower_circuit_limit,
            upper_circuit=request.upper_circuit_limit,
        )

        # Verify Clause 1.6 on both legs vs CMP
        c16_sl_ok, c16_sl_err = self.check_clause_1_6(geometry.sl_trigger, request.cmp)
        if not c16_sl_ok and c16_sl_err:
            rejection_reasons.append(RejectionReason.CLAUSE_1_6_PROXIMITY_VIOLATION)
            rejection_details.append(f"SL leg: {c16_sl_err}")

        c16_tgt_ok, c16_tgt_err = self.check_clause_1_6(geometry.target_trigger, request.cmp)
        if not c16_tgt_ok and c16_tgt_err:
            rejection_reasons.append(RejectionReason.CLAUSE_1_6_PROXIMITY_VIOLATION)
            rejection_details.append(f"Target leg: {c16_tgt_err}")

        # Circuit limit checks
        circ_ok, circ_errs = self.check_circuit_limits(
            {
                "sl_trigger": geometry.sl_trigger,
                "sl_limit": geometry.sl_limit,
                "target_trigger": geometry.target_trigger,
                "target_limit": geometry.target_limit,
            },
            request.lower_circuit_limit,
            request.upper_circuit_limit,
        )
        if not circ_ok:
            rejection_reasons.append(RejectionReason.CIRCUIT_LIMIT_BREACH)
            rejection_details.extend(circ_errs)

        is_valid = len(rejection_reasons) == 0

        proposal_payload: dict[str, Any] | None = None
        if is_valid:
            self.record_proposal(request.tradingsymbol, request.stage, request.custom_timestamp)
            proposal_payload = {
                "type": "two-leg",
                "condition": {
                    "exchange": "NSE",
                    "tradingsymbol": request.tradingsymbol.upper(),
                    "trigger_values": [geometry.sl_trigger, geometry.target_trigger],
                    "last_price": request.cmp,
                },
                "orders": [
                    {
                        "exchange": "NSE",
                        "tradingsymbol": request.tradingsymbol.upper(),
                        "transaction_type": "SELL",
                        "quantity": sell_qty,
                        "order_type": "LIMIT",
                        "product": "CNC",
                        "price": geometry.sl_limit,
                    },
                    {
                        "exchange": "NSE",
                        "tradingsymbol": request.tradingsymbol.upper(),
                        "transaction_type": "SELL",
                        "quantity": sell_qty,
                        "order_type": "LIMIT",
                        "product": "CNC",
                        "price": geometry.target_limit,
                    },
                ],
                "stage": request.stage.value,
            }

        return RiskValidationResult(
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            rejection_details=rejection_details,
            geometry=geometry,
            sizing=None,
            proposal_payload=proposal_payload,
            diagnostics=diagnostics,
        )
