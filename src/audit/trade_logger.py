"""Structured JSONL trade and GTT order audit logger.

Maintains an immutable append-only JSONL log file with chmod 600 permissions
for complete compliance, auditability, and trade verification.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import structlog

logger = structlog.get_logger(__name__)


class TradeAuditLogger:
    """Audit logger appending structured events to a chmod 600 JSONL file."""

    def __init__(self, log_path: str | Path = "data/trade_audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_secure_permissions()

    def _ensure_secure_permissions(self) -> None:
        """Enforce chmod 600 on file if it exists, or create it with 600."""
        if not self.log_path.exists():
            # Create file
            self.log_path.touch()

        # On POSIX systems, apply chmod 600 (owner read/write only)
        if sys.platform != "win32":
            try:
                os.chmod(self.log_path, 0o600)
            except OSError as e:
                logger.warning("chmod_failed", path=str(self.log_path), error=str(e))

    def log_event(
        self,
        event_type: str,
        tradingsymbol: str,
        action: str,
        stage: int,
        dry_run: bool,
        details: Mapping[str, Any] | None = None,
        trigger_id: int | None = None,
    ) -> dict[str, Any]:
        """Append a structured trade audit event to the JSONL log.

        Args:
            event_type: E.g., 'GTT_PROPOSED', 'GTT_APPROVED', 'GTT_SUBMITTED', 'GTT_DRY_RUN'.
            tradingsymbol: NSE symbol (e.g., 'RELIANCE', 'TCS').
            action: BUY, SELL_OCO, MODIFY, DELETE.
            stage: 1 (Dip Entry) or 2 (Post-fill OCO SL/Target).
            dry_run: Whether execution was simulated or live.
            details: Extra metadata (price, limit, qty, thesis, etc.).
            trigger_id: Zerodha GTT trigger ID (integer).

        Returns:
            The complete logged event dict.
        """
        self._ensure_secure_permissions()

        event_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type.upper(),
            "tradingsymbol": tradingsymbol.upper(),
            "action": action.upper(),
            "stage": int(stage),
            "dry_run": bool(dry_run),
            "trigger_id": int(trigger_id) if trigger_id is not None else None,
            "details": dict(details or {}),
        }

        line = json.dumps(event_payload, default=str) + "\n"

        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line)

        logger.info(
            "trade_event_audited",
            event_type=event_payload["event_type"],
            tradingsymbol=event_payload["tradingsymbol"],
            dry_run=event_payload["dry_run"],
        )

        return event_payload

    def read_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read the most recent audit events."""
        if not self.log_path.exists():
            return []

        events: list[dict[str, Any]] = []
        try:
            with open(self.log_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as e:
            logger.error("read_audit_log_failed", error=str(e))
            return []

        return events[-limit:]
