"""Async scheduling engine coordinating positional trading routines in IST."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import structlog

from src.broker.gtt_executor import GTTExecutor
from src.broker.kite_mcp_client import KiteMCPClient
from src.config.settings import Settings, get_settings
from src.risk.risk_guard import RiskGuard
from src.scheduler.alerts import AlertDispatcher, AlertSeverity
from src.scheduler.jobs import (
    IST,
    JobExecutionResult,
    run_eod_bhavcopy_dip_scan_job,
    run_post_cas_reconciliation_job,
    run_pre_market_job,
    run_weekly_digest_job,
)
from src.storage.ledger import SQLiteLedger
from src.technicals.dip_scanner import TechnicalDipScanner

logger = structlog.get_logger(__name__)


def compute_next_run(
    target_time_str: str,
    target_weekday: int | None = None,
    current_dt: datetime | None = None,
) -> datetime:
    """Compute next occurrence of target IST time (and optional weekday: 0=Mon, 4=Fri)."""
    now = current_dt or datetime.now(IST)
    target_h, target_m = map(int, target_time_str.split(":"))
    cand = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)

    if target_weekday is not None:
        # Match weekday first
        days_ahead = (target_weekday - now.weekday()) % 7
        cand = cand + timedelta(days=days_ahead)
        if cand <= now:
            cand = cand + timedelta(days=7)
    else:
        if cand <= now:
            cand = cand + timedelta(days=1)

    return cand


class JobScheduler:
    """Orchestrates scheduled positional jobs for Zerodha CNC dip-investing."""

    def __init__(
        self,
        settings: Settings | None = None,
        ledger: SQLiteLedger | None = None,
        kite_client: KiteMCPClient | None = None,
        gtt_executor: GTTExecutor | None = None,
        risk_guard: RiskGuard | None = None,
        dip_scanner: TechnicalDipScanner | None = None,
        alert_dispatcher: AlertDispatcher | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ledger = ledger or SQLiteLedger(self.settings.db_path)
        self.kite_client = kite_client
        self.executor = gtt_executor or GTTExecutor(
            kite_client=self.kite_client,
            ledger=self.ledger,
            settings=self.settings,
        )
        self.risk_guard = risk_guard or RiskGuard()
        self.dip_scanner = dip_scanner or TechnicalDipScanner()
        self.alerts = alert_dispatcher or AlertDispatcher(self.settings)

        self._running = False
        self._execution_history: list[JobExecutionResult] = []

    async def execute_job(self, job_name: str, **kwargs: Any) -> JobExecutionResult:
        """Execute a specific job by name with error boundary protection."""
        logger.info("executing_scheduled_job", job_name=job_name)
        try:
            if job_name in ("pre_market", "pre_market_job"):
                res = await run_pre_market_job(
                    ledger=self.ledger,
                    kite_client=self.kite_client,
                    settings=self.settings,
                    alert_dispatcher=self.alerts,
                    **kwargs,
                )
            elif job_name in ("post_cas", "post_cas_reconciliation_job"):
                res = await run_post_cas_reconciliation_job(
                    ledger=self.ledger,
                    kite_client=self.kite_client,
                    gtt_executor=self.executor,
                    risk_guard=self.risk_guard,
                    settings=self.settings,
                    alert_dispatcher=self.alerts,
                    **kwargs,
                )
            elif job_name in ("eod_bhavcopy", "eod_bhavcopy_dip_scan_job"):
                res = await run_eod_bhavcopy_dip_scan_job(
                    ledger=self.ledger,
                    dip_scanner=self.dip_scanner,
                    settings=self.settings,
                    alert_dispatcher=self.alerts,
                    **kwargs,
                )
            elif job_name in ("weekly_digest", "weekly_digest_job"):
                res = await run_weekly_digest_job(
                    ledger=self.ledger,
                    settings=self.settings,
                    alert_dispatcher=self.alerts,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown job name: {job_name}")

            self._execution_history.append(res)
            return res

        except Exception as e:
            logger.error("job_execution_exception", job_name=job_name, error=str(e), exc_info=True)
            iso_now = datetime.now(IST).isoformat()
            alert = await self.alerts.dispatch(
                title=f"Scheduled Job Failed: {job_name}",
                message=f"Exception raised during scheduled routine: {e}",
                severity=AlertSeverity.CRITICAL,
                metadata={"job_name": job_name, "error": str(e)},
            )
            res = JobExecutionResult(
                job_name=job_name,
                success=False,
                timestamp=iso_now,
                summary=f"Job failed with error: {e}",
                details={"error": str(e)},
                alerts_dispatched=[alert],
            )
            self._execution_history.append(res)
            return res

    def get_history(self) -> list[JobExecutionResult]:
        """Return history of executed jobs."""
        return list(self._execution_history)
