"""Alert dispatching and notification system for positional dip-investing jobs.

Supports:
1. Multi-channel alerting (Telegram Bot, Slack Webhook, and Structured Logger).
2. Severity classification: INFO, WARNING, CRITICAL (e.g. gap-down unfilled SL).
3. Resilient network handling: external API errors or unconfigured webhooks do not crash jobs.
4. Dry-run and simulation mode: captures alert history in-memory for testing and offline audits.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import httpx
import structlog

from src.config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Severity levels for operational and trading alerts."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class AlertMessage:
    """Standardized alert notification object."""

    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.INFO
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def formatted_text(self) -> str:
        """Format a clear plain-text message for external messengers."""
        prefix = {
            AlertSeverity.INFO: "ℹ️ [INFO]",
            AlertSeverity.WARNING: "⚠️ [WARNING]",
            AlertSeverity.CRITICAL: "🚨 [CRITICAL ALERT]",
        }.get(self.severity, "[ALERT]")
        return f"{prefix} *{self.title}*\n\n{self.message}"


class AlertDispatcher:
    """Dispatches operational alerts to configured external webhooks and structured logs."""

    def __init__(
        self,
        settings: Settings | None = None,
        dry_run: bool | None = None,
        timeout_sec: float = 10.0,
    ) -> None:
        self.settings = settings or get_settings()
        self.dry_run = dry_run if dry_run is not None else self.settings.dry_run
        self.timeout_sec = timeout_sec
        self.dispatched_alerts: list[AlertMessage] = []

    async def _send_slack(self, client: httpx.AsyncClient, alert: AlertMessage) -> bool:
        """Send formatted alert to Slack incoming webhook."""
        webhook_url = self.settings.slack_webhook_url
        if not webhook_url:
            return False

        header_prefix = {
            AlertSeverity.INFO: "ℹ️ ",
            AlertSeverity.WARNING: "⚠️ ",
            AlertSeverity.CRITICAL: "🚨 CRITICAL: ",
        }.get(alert.severity, "")

        payload = {
            "text": alert.formatted_text(),
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{header_prefix}{alert.title}"[:150],
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": alert.message[:2900],
                    },
                },
            ],
        }

        try:
            resp = await client.post(webhook_url, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            logger.info("slack_alert_sent", title=alert.title, status_code=resp.status_code)
            return True
        except Exception as e:
            logger.warning("slack_alert_send_failed", title=alert.title, error=str(e))
            return False

    async def _send_telegram(self, client: httpx.AsyncClient, alert: AlertMessage) -> bool:
        """Send formatted alert to Telegram chat via Bot API."""
        token = self.settings.telegram_bot_token
        chat_id = self.settings.telegram_chat_id
        if not token or not chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": alert.formatted_text(),
            "parse_mode": "Markdown",
        }

        try:
            resp = await client.post(url, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            logger.info("telegram_alert_sent", title=alert.title, status_code=resp.status_code)
            return True
        except Exception as e:
            logger.warning("telegram_alert_send_failed", title=alert.title, error=str(e))
            return False

    async def dispatch(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> AlertMessage:
        """Dispatch an operational alert to all configured targets."""
        alert = AlertMessage(
            title=title,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )

        # 1. Always record in-memory for testing / auditing
        self.dispatched_alerts.append(alert)

        # 2. Structured logging
        log_ctx = {
            "title": alert.title,
            "severity": alert.severity.value,
            "timestamp": alert.timestamp,
            "dry_run": self.dry_run,
            **alert.metadata,
        }
        if severity == AlertSeverity.CRITICAL:
            logger.error("operational_alert_critical", **log_ctx)
        elif severity == AlertSeverity.WARNING:
            logger.warning("operational_alert_warning", **log_ctx)
        else:
            logger.info("operational_alert_info", **log_ctx)

        # 3. External delivery if not in dry-run mode
        if not self.dry_run:
            async with httpx.AsyncClient() as client:
                await asyncio.gather(
                    self._send_slack(client, alert),
                    self._send_telegram(client, alert),
                    return_exceptions=True,
                )

        return alert

    def dispatch_sync(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> AlertMessage:
        """Synchronous wrapper for dispatch."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already inside an async loop, schedule as task
            alert = AlertMessage(title=title, message=message, severity=severity, metadata=metadata or {})
            self.dispatched_alerts.append(alert)
            asyncio.create_task(self.dispatch(title, message, severity, metadata))
            return alert
        else:
            return asyncio.run(self.dispatch(title, message, severity, metadata))

    def clear(self) -> None:
        """Clear recorded alerts history."""
        self.dispatched_alerts.clear()
