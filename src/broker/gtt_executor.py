"""Deterministic and Safe GTT Execution Engine.

Handles:
1. GTT-Only Brokerage Scope (never calls place_order, modify_order, or cancel_order).
2. Safe Place-Verify-Delete GTT modification pattern (place new -> verify in get_gtts -> delete old).
3. Dry-Run safety default (DRY_RUN=True prevents real MCP writes).
4. Automatic dual ledger and chmod 600 JSONL audit trail logging.
5. Pending-Buy cash commitment reservation and release.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
import re
from typing import Any

import structlog

from src.audit.trade_logger import TradeAuditLogger
from src.broker.kite_mcp_client import KiteMCPClient
from src.config.settings import Settings, load_settings
from src.storage.ledger import SQLiteLedger

logger = structlog.get_logger(__name__)


class GTTExecutionError(Exception):
    """Raised when GTT execution fails."""


class GTTVerificationError(GTTExecutionError):
    """Raised when newly placed GTT fails existence verification in get_gtts."""


@dataclass
class GTTOrderPayload:
    """Standardized GTT order payload."""

    tradingsymbol: str
    exchange: str = "NSE"
    transaction_type: str = "BUY"  # "BUY" or "SELL"
    trigger_type: str = "single"  # "single" or "two-leg"
    trigger_values: list[float] = field(default_factory=list)
    limit_prices: list[float] = field(default_factory=list)
    quantities: list[int] = field(default_factory=list)
    product: str = "CNC"  # strictly CNC delivery
    stage: int = 1  # 1 (Dip Entry) or 2 (Post-fill OCO)
    thesis: str | None = None
    allocated_cash: float = 0.0


@dataclass
class GTTExecutionResult:
    """Outcome of a GTT order operation."""

    success: bool
    action: str  # "PLACE", "MODIFY_PLACE_VERIFY_DELETE", "DELETE", "REJECTED"
    stage: int
    tradingsymbol: str
    trigger_id: int | None = None
    old_trigger_id: int | None = None
    verified: bool = False
    dry_run: bool = True
    status: str = "COMPLETED"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def extract_trigger_id(resp: Any) -> int | None:
    """Extract integer trigger_id from MCP or raw response."""
    if resp is None:
        return None
    if isinstance(resp, int):
        return resp
    if isinstance(resp, str):
        try:
            return int(resp)
        except ValueError:
            pass
        match = re.search(r"['\"]?trigger_id['\"]?\s*:\s*(\d+)", resp)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d{7,10})\b", resp)
        if match:
            return int(match.group(1))
    if isinstance(resp, dict):
        if "trigger_id" in resp:
            try:
                return int(resp["trigger_id"])
            except (ValueError, TypeError):
                pass
        if "data" in resp and isinstance(resp["data"], dict) and "trigger_id" in resp["data"]:
            try:
                return int(resp["data"]["trigger_id"])
            except (ValueError, TypeError):
                pass
    return None


class GTTExecutor:
    """Executes GTT order operations strictly conforming to risk rules and SEBI safety boundaries."""

    def __init__(
        self,
        kite_client: KiteMCPClient | None = None,
        ledger: SQLiteLedger | None = None,
        trade_logger: TradeAuditLogger | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.kite_client = kite_client
        self.ledger = ledger or SQLiteLedger()
        self.trade_logger = trade_logger or TradeAuditLogger()
        self.settings = settings or load_settings()

    async def place_gtt(self, payload: GTTOrderPayload) -> GTTExecutionResult:
        """Place a single-leg or two-leg GTT order."""
        if payload.product.upper() != "CNC":
            raise ValueError(f"Strict CNC policy: GTT product must be 'CNC', got {payload.product!r}")

        is_dry_run = bool(self.settings.dry_run)
        symbol = payload.tradingsymbol.upper()
        trigger_price = payload.trigger_values[0] if payload.trigger_values else 0.0
        limit_price = payload.limit_prices[0] if payload.limit_prices else 0.0
        qty = payload.quantities[0] if payload.quantities else 0

        logger.info(
            "gtt_place_requested",
            symbol=symbol,
            stage=payload.stage,
            dry_run=is_dry_run,
            trigger_price=trigger_price,
            limit_price=limit_price,
            quantity=qty,
        )

        if is_dry_run:
            mock_id = random.randint(70000000, 79999999)
            details = {
                "exchange": payload.exchange,
                "trigger_values": payload.trigger_values,
                "limit_prices": payload.limit_prices,
                "quantities": payload.quantities,
                "product": payload.product,
                "thesis": payload.thesis,
            }

            self.trade_logger.log_event(
                event_type="GTT_DRY_RUN",
                tradingsymbol=symbol,
                action="BUY" if payload.stage == 1 else "SELL_OCO",
                stage=payload.stage,
                dry_run=True,
                trigger_id=mock_id,
                details=details,
            )

            self.ledger.log_gtt_action(
                trigger_id=mock_id,
                action="BUY" if payload.stage == 1 else "SELL_OCO",
                stage=payload.stage,
                tradingsymbol=symbol,
                trigger_price=trigger_price,
                limit_price=limit_price,
                quantity=qty,
                status="DRY_RUN_PLACED",
                request_payload=details,
                response_payload={"trigger_id": mock_id, "mock": True},
            )

            if payload.stage == 1:
                allocated = payload.allocated_cash or (limit_price * qty)
                self.ledger.commit_pending_buy_cash(
                    tradingsymbol=symbol,
                    allocated_cash=allocated,
                    trigger_id=mock_id,
                )

            return GTTExecutionResult(
                success=True,
                action="PLACE",
                stage=payload.stage,
                tradingsymbol=symbol,
                trigger_id=mock_id,
                verified=True,
                dry_run=True,
                status="DRY_RUN_PLACED",
                message=f"[DRY RUN] Simulated GTT placed with trigger_id={mock_id}",
                details=details,
            )

        # Live execution via Kite MCP
        if not self.kite_client:
            raise GTTExecutionError("KiteMCPClient required for live GTT placement.")

        resp = await self.kite_client.place_gtt_order(
            tradingsymbol=symbol,
            exchange=payload.exchange,
            transaction_type=payload.transaction_type,
            trigger_type=payload.trigger_type,
            trigger_values=payload.trigger_values,
            limit_prices=payload.limit_prices,
            quantities=payload.quantities,
            product="CNC",
        )

        trigger_id = extract_trigger_id(resp)
        if not trigger_id:
            logger.error("gtt_place_missing_trigger_id", response=resp)
            raise GTTExecutionError(f"GTT placement response missing trigger_id: {resp}")

        details = {
            "exchange": payload.exchange,
            "trigger_values": payload.trigger_values,
            "limit_prices": payload.limit_prices,
            "quantities": payload.quantities,
            "product": "CNC",
            "thesis": payload.thesis,
            "broker_response": resp,
        }

        self.trade_logger.log_event(
            event_type="GTT_SUBMITTED",
            tradingsymbol=symbol,
            action="BUY" if payload.stage == 1 else "SELL_OCO",
            stage=payload.stage,
            dry_run=False,
            trigger_id=trigger_id,
            details=details,
        )

        self.ledger.log_gtt_action(
            trigger_id=trigger_id,
            action="BUY" if payload.stage == 1 else "SELL_OCO",
            stage=payload.stage,
            tradingsymbol=symbol,
            trigger_price=trigger_price,
            limit_price=limit_price,
            quantity=qty,
            status="PLACED",
            request_payload=details,
            response_payload=resp if isinstance(resp, dict) else {"raw": str(resp)},
        )

        if payload.stage == 1:
            allocated = payload.allocated_cash or (limit_price * qty)
            self.ledger.commit_pending_buy_cash(
                tradingsymbol=symbol,
                allocated_cash=allocated,
                trigger_id=trigger_id,
            )

        return GTTExecutionResult(
            success=True,
            action="PLACE",
            stage=payload.stage,
            tradingsymbol=symbol,
            trigger_id=trigger_id,
            verified=True,
            dry_run=False,
            status="PLACED",
            message=f"Live GTT successfully placed with trigger_id={trigger_id}",
            details=details,
        )

    async def verify_gtt_exists(self, trigger_id: int, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
        """Verify that a newly placed GTT order appears in get_gtts active list."""
        if self.settings.dry_run or not self.kite_client:
            return True

        for attempt in range(1, max_retries + 1):
            try:
                gtts = await self.kite_client.get_gtts()
                if isinstance(gtts, list):
                    for g in gtts:
                        if isinstance(g, dict):
                            g_id = g.get("id") or g.get("trigger_id")
                            if g_id is not None and int(g_id) == int(trigger_id):
                                logger.info("gtt_verification_succeeded", trigger_id=trigger_id, attempt=attempt)
                                return True
            except Exception as e:
                logger.warning("gtt_verification_probe_error", trigger_id=trigger_id, error=str(e), attempt=attempt)

            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

        return False

    async def modify_gtt_place_verify_delete(
        self,
        old_trigger_id: int,
        payload: GTTOrderPayload,
    ) -> GTTExecutionResult:
        """Safely modify a GTT order using Place-Verify-Delete.

        Step 1: Place new GTT order.
        Step 2: Verify new GTT order exists in get_gtts().
        Step 3: Only if verified, delete the old GTT order.
        If verification fails, old GTT is NEVER deleted.
        """
        if not isinstance(old_trigger_id, int):
            old_trigger_id = int(old_trigger_id)

        logger.info(
            "gtt_modify_pvd_started",
            old_trigger_id=old_trigger_id,
            symbol=payload.tradingsymbol,
            dry_run=self.settings.dry_run,
        )

        # Step 1: Place new GTT
        place_res = await self.place_gtt(payload)
        new_trigger_id = place_res.trigger_id
        if not new_trigger_id:
            raise GTTExecutionError(f"Place step failed during GTT modification for {payload.tradingsymbol}")

        # Step 2: Verify new GTT in get_gtts
        is_verified = await self.verify_gtt_exists(new_trigger_id)
        if not is_verified:
            msg = (
                f"CRITICAL SAFETY ABORT: New GTT {new_trigger_id} could not be verified in get_gtts. "
                f"Old GTT {old_trigger_id} has been PRESERVED to maintain position protection."
            )
            logger.error(
                "gtt_pvd_verification_failed",
                new_trigger_id=new_trigger_id,
                old_trigger_id=old_trigger_id,
            )
            raise GTTVerificationError(msg)

        # Step 3: Delete old GTT
        await self.delete_gtt(
            trigger_id=old_trigger_id,
            tradingsymbol=payload.tradingsymbol,
            stage=payload.stage,
            reason=f"Replaced by new GTT {new_trigger_id} via Place-Verify-Delete",
        )

        # Release old cash reservation if stage 1
        if payload.stage == 1:
            self.ledger.release_pending_buy_cash(trigger_id=old_trigger_id)

        return GTTExecutionResult(
            success=True,
            action="MODIFY_PLACE_VERIFY_DELETE",
            stage=payload.stage,
            tradingsymbol=payload.tradingsymbol,
            trigger_id=new_trigger_id,
            old_trigger_id=old_trigger_id,
            verified=True,
            dry_run=place_res.dry_run,
            status="MODIFIED",
            message=f"Safe Place-Verify-Delete completed: Placed #{new_trigger_id}, Verified, Deleted #{old_trigger_id}",
            details={
                "new_order": place_res.details,
                "deleted_old_id": old_trigger_id,
            },
        )

    async def delete_gtt(
        self,
        trigger_id: int,
        tradingsymbol: str,
        stage: int = 1,
        reason: str | None = None,
    ) -> GTTExecutionResult:
        """Delete an active GTT order."""
        if not isinstance(trigger_id, int):
            trigger_id = int(trigger_id)

        is_dry_run = bool(self.settings.dry_run)
        symbol = tradingsymbol.upper()

        logger.info(
            "gtt_delete_requested",
            trigger_id=trigger_id,
            symbol=symbol,
            dry_run=is_dry_run,
            reason=reason,
        )

        details = {"reason": reason, "deleted_at": datetime.now(timezone.utc).isoformat()}

        if is_dry_run:
            self.trade_logger.log_event(
                event_type="GTT_DRY_RUN",
                tradingsymbol=symbol,
                action="DELETE",
                stage=stage,
                dry_run=True,
                trigger_id=trigger_id,
                details=details,
            )
            self.ledger.log_gtt_action(
                trigger_id=trigger_id,
                action="DELETE",
                stage=stage,
                tradingsymbol=symbol,
                trigger_price=0.0,
                limit_price=0.0,
                quantity=0,
                status="DRY_RUN_DELETED",
                request_payload=details,
            )
            if stage == 1:
                self.ledger.release_pending_buy_cash(trigger_id=trigger_id)

            return GTTExecutionResult(
                success=True,
                action="DELETE",
                stage=stage,
                tradingsymbol=symbol,
                trigger_id=trigger_id,
                dry_run=True,
                status="DRY_RUN_DELETED",
                message=f"[DRY RUN] Simulated deletion of GTT #{trigger_id}",
                details=details,
            )

        if not self.kite_client:
            raise GTTExecutionError("KiteMCPClient required for live GTT deletion.")

        resp = await self.kite_client.delete_gtt_order(trigger_id=trigger_id)

        self.trade_logger.log_event(
            event_type="GTT_DELETED",
            tradingsymbol=symbol,
            action="DELETE",
            stage=stage,
            dry_run=False,
            trigger_id=trigger_id,
            details={"broker_response": resp, **details},
        )
        self.ledger.log_gtt_action(
            trigger_id=trigger_id,
            action="DELETE",
            stage=stage,
            tradingsymbol=symbol,
            trigger_price=0.0,
            limit_price=0.0,
            quantity=0,
            status="DELETED",
            request_payload=details,
            response_payload=resp if isinstance(resp, dict) else {"raw": str(resp)},
        )
        if stage == 1:
            self.ledger.release_pending_buy_cash(trigger_id=trigger_id)

        return GTTExecutionResult(
            success=True,
            action="DELETE",
            stage=stage,
            tradingsymbol=symbol,
            trigger_id=trigger_id,
            dry_run=False,
            status="DELETED",
            message=f"Live GTT #{trigger_id} deleted successfully",
            details=details,
        )
