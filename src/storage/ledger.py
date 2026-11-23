"""Durable SQLite trade and cache ledger for positional CNC investing.

Manages:
1. positions_ledger: Track filled CNC positions, entry dates/prices, STCG/LTCG holding period, stop-loss and targets.
2. gtt_audit_log: Log all GTT orders, triggers, modifications and deletions with strict integer trigger_id.
3. pending_buy_commits: Cash reservations for active Stage 1 Buy GTTs so cash is not double-allocated.
4. screener_cache: Quarterly-cached Screener.in fundamental ratios (30-90 day TTL).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import structlog

logger = structlog.get_logger(__name__)


class SQLiteLedger:
    """Thread-safe SQLite database manager for trade state and caches."""

    def __init__(self, db_path: str | Path = "data/copilot.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tradingsymbol TEXT NOT NULL,
                    isin TEXT,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    current_stop_loss REAL NOT NULL,
                    target_price REAL NOT NULL,
                    thesis TEXT,
                    stage TEXT NOT NULL DEFAULT 'STAGE_1_PENDING',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_positions_symbol 
                ON positions_ledger (tradingsymbol);

                CREATE TABLE IF NOT EXISTS gtt_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger_id INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    stage INTEGER NOT NULL,
                    tradingsymbol TEXT NOT NULL,
                    trigger_price REAL NOT NULL,
                    limit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request_payload TEXT,
                    response_payload TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_gtt_trigger_id 
                ON gtt_audit_log (trigger_id);

                CREATE TABLE IF NOT EXISTS pending_buy_commits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tradingsymbol TEXT NOT NULL,
                    allocated_cash REAL NOT NULL,
                    trigger_id INTEGER,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS screener_cache (
                    symbol TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    ttl_days INTEGER NOT NULL DEFAULT 60
                );
                """
            )
        logger.info("sqlite_ledger_initialized", db_path=str(self.db_path))

    # --- positions_ledger methods ---

    def record_position(
        self,
        tradingsymbol: str,
        entry_price: float,
        quantity: int,
        current_stop_loss: float,
        target_price: float,
        isin: str | None = None,
        entry_date: str | None = None,
        thesis: str | None = None,
        stage: str = "STAGE_2_ARMED",
    ) -> int:
        """Record or create a position in the ledger."""
        now = datetime.now(timezone.utc).isoformat()
        edate = entry_date or datetime.now().strftime("%Y-%m-%d")

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO positions_ledger (
                    tradingsymbol, isin, entry_date, entry_price, quantity,
                    current_stop_loss, target_price, thesis, stage,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tradingsymbol.upper(),
                    isin,
                    edate,
                    float(entry_price),
                    int(quantity),
                    float(current_stop_loss),
                    float(target_price),
                    thesis,
                    stage,
                    now,
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch all active / armed positions."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM positions_ledger 
                WHERE stage != 'CLOSED' 
                ORDER BY created_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def update_position_stop(self, position_id: int, new_stop_loss: float) -> bool:
        """Update the trailing or initial stop-loss of a position."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE positions_ledger 
                SET current_stop_loss = ?, updated_at = ? 
                WHERE id = ?
                """,
                (float(new_stop_loss), now, position_id),
            )
            return cursor.rowcount > 0

    def close_position(self, position_id: int) -> bool:
        """Mark a position as closed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE positions_ledger SET stage = 'CLOSED', updated_at = ? WHERE id = ?",
                (now, position_id),
            )
            return cursor.rowcount > 0

    # --- gtt_audit_log methods ---

    def log_gtt_action(
        self,
        trigger_id: int,
        action: str,
        stage: int,
        tradingsymbol: str,
        trigger_price: float,
        limit_price: float,
        quantity: int,
        status: str,
        request_payload: dict[str, Any] | None = None,
        response_payload: dict[str, Any] | None = None,
    ) -> int:
        """Record a GTT operation with strictly enforced integer trigger_id."""
        if not isinstance(trigger_id, int):
            try:
                trigger_id = int(trigger_id)
            except (ValueError, TypeError) as e:
                raise ValueError(f"trigger_id must be strictly an integer, got {trigger_id!r}") from e

        now = datetime.now(timezone.utc).isoformat()
        req_json = json.dumps(request_payload) if request_payload else None
        res_json = json.dumps(response_payload) if response_payload else None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO gtt_audit_log (
                    trigger_id, action, stage, tradingsymbol,
                    trigger_price, limit_price, quantity, status,
                    request_payload, response_payload, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger_id,
                    action.upper(),
                    int(stage),
                    tradingsymbol.upper(),
                    float(trigger_price),
                    float(limit_price),
                    int(quantity),
                    status.upper(),
                    req_json,
                    res_json,
                    now,
                ),
            )
            return cursor.lastrowid  # type: ignore

    def get_gtt_audit_history(self, tradingsymbol: str | None = None) -> list[dict[str, Any]]:
        """Retrieve GTT audit logs optionally filtered by symbol."""
        with self._get_connection() as conn:
            if tradingsymbol:
                rows = conn.execute(
                    "SELECT * FROM gtt_audit_log WHERE tradingsymbol = ? ORDER BY id DESC",
                    (tradingsymbol.upper(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM gtt_audit_log ORDER BY id DESC").fetchall()
            return [dict(row) for row in rows]

    # --- pending_buy_commits methods (cash reservation) ---

    def commit_pending_buy_cash(
        self,
        tradingsymbol: str,
        allocated_cash: float,
        trigger_id: int | None = None,
        expires_at: str | None = None,
    ) -> int:
        """Reserve cash for an active Stage 1 Buy GTT."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pending_buy_commits (
                    tradingsymbol, allocated_cash, trigger_id, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (tradingsymbol.upper(), float(allocated_cash), trigger_id, now, expires_at),
            )
            return cursor.lastrowid  # type: ignore

    def get_total_pending_buy_cash(self) -> float:
        """Calculate total cash currently reserved for pending Buy GTTs."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(allocated_cash), 0.0) as total FROM pending_buy_commits"
            ).fetchone()
            return float(row["total"]) if row else 0.0

    def get_pending_buy_commits(self, tradingsymbol: str | None = None) -> list[dict[str, Any]]:
        """Retrieve list of active pending buy commitments."""
        with self._get_connection() as conn:
            if tradingsymbol:
                rows = conn.execute(
                    "SELECT * FROM pending_buy_commits WHERE tradingsymbol = ? ORDER BY id DESC",
                    (tradingsymbol.upper(),),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM pending_buy_commits ORDER BY id DESC").fetchall()
            return [dict(row) for row in rows]

    def release_pending_buy_cash(
        self,
        commit_id: int | None = None,
        trigger_id: int | None = None,
        tradingsymbol: str | None = None,
    ) -> int:
        """Release reserved cash when a Buy GTT is filled or cancelled."""
        with self._get_connection() as conn:
            if commit_id is not None:
                cursor = conn.execute("DELETE FROM pending_buy_commits WHERE id = ?", (commit_id,))
            elif trigger_id is not None:
                cursor = conn.execute("DELETE FROM pending_buy_commits WHERE trigger_id = ?", (trigger_id,))
            elif tradingsymbol is not None:
                cursor = conn.execute("DELETE FROM pending_buy_commits WHERE tradingsymbol = ?", (tradingsymbol.upper(),))
            else:
                return 0
            return cursor.rowcount

    # --- screener_cache methods ---

    def get_screener_cache(self, symbol: str) -> dict[str, Any] | None:
        """Get cached Screener data if not expired."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at, ttl_days FROM screener_cache WHERE symbol = ?",
                (symbol.upper(),),
            ).fetchone()
            if not row:
                return None

            fetched_at = datetime.fromisoformat(row["fetched_at"])
            age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400.0
            if age_days > row["ttl_days"]:
                logger.info("screener_cache_expired", symbol=symbol, age_days=round(age_days, 1))
                return None

            try:
                return json.loads(row["data_json"])
            except json.JSONDecodeError:
                return None

    def set_screener_cache(self, symbol: str, data: dict[str, Any], ttl_days: int = 60) -> None:
        """Store Screener.in fundamentals in cache with quarterly TTL."""
        now = datetime.now(timezone.utc).isoformat()
        data_json = json.dumps(data)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO screener_cache (symbol, data_json, fetched_at, ttl_days)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    data_json = excluded.data_json,
                    fetched_at = excluded.fetched_at,
                    ttl_days = excluded.ttl_days
                """,
                (symbol.upper(), data_json, now, int(ttl_days)),
            )
