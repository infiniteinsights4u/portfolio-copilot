"""Durable SQLite Checkpointer for LangGraph.

Implements BaseCheckpointSaver backed by SQLite with WAL mode enabled.
Persists graph checkpoints, channel values (blobs), and pending writes
across thread runs and process restarts without InMemorySaver or external services.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
import random
import sqlite3
from types import TracebackType
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = structlog.get_logger(__name__)

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB NOT NULL,
    metadata_type TEXT,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BLOB,
    task_path TEXT DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_parent 
ON checkpoints (thread_id, checkpoint_ns, parent_checkpoint_id);
"""


DEFAULT_MSGPACK_ALLOWLIST = [
    ("src.risk.risk_guard", "RejectionReason"),
    ("src.risk.risk_guard", "OrderGeometry"),
    ("src.risk.risk_guard", "SizingResult"),
    ("src.risk.risk_guard", "RiskValidationResult"),
    ("src.broker.gtt_executor", "GTTExecutionResult"),
    ("src.broker.gtt_executor", "GTTOrderPayload"),
]


class DurableSqliteSaver(
    BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager
):
    """Durable SQLite checkpointer for LangGraph workflows.

    Persists checkpoints to a local SQLite database file or in-memory SQLite db.
    Thread-safe and WAL-mode enabled.
    """

    def __init__(
        self,
        db_path: str | Path = "portfolio_copilot_checkpoints.db",
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        default_serde = JsonPlusSerializer().with_msgpack_allowlist(DEFAULT_MSGPACK_ALLOWLIST)
        super().__init__(serde=serde or default_serde)
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._async_lock: asyncio.Lock | None = None
        self._init_db()

    @property
    def _lock(self) -> asyncio.Lock:
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                autocommit=True,
            )
            self._conn.row_factory = sqlite3.Row
            if self.db_path != ":memory:":
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                    self._conn.execute("PRAGMA synchronous=NORMAL;")
                except Exception:
                    pass
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.executescript(CREATE_TABLES_SQL)

    def __enter__(self) -> DurableSqliteSaver:
        self._get_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return None

    async def __aenter__(self) -> DurableSqliteSaver:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self.__exit__(exc_type, exc_value, traceback)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if not versions:
            return result

        conn = self._get_connection()
        for k, ver in versions.items():
            row = conn.execute(
                """
                SELECT type, blob FROM checkpoint_blobs
                WHERE thread_id = ? AND checkpoint_ns = ? AND channel = ? AND version = ?
                """,
                (thread_id, checkpoint_ns, k, str(ver)),
            ).fetchone()

            if row is not None:
                blob_type = row["type"]
                blob_bytes = row["blob"]
                if blob_type != "empty":
                    result[k] = self.serde.loads_typed((blob_type, blob_bytes))
        return result

    def _ordered_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, tuple[str, bytes], str]]:
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT task_id, idx, channel, type, blob, task_path
            FROM checkpoint_writes
            WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
            """,
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()

        sorted_rows = sorted(
            rows,
            key=lambda r: writes_sort_key(r["task_path"] or "", r["task_id"], r["idx"]),
        )

        parsed: list[tuple[str, str, tuple[str, bytes], str]] = []
        for r in sorted_rows:
            parsed.append((
                r["task_id"],
                r["channel"],
                (r["type"], r["blob"]),
                r["task_path"] or "",
            ))
        return parsed

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the SQLite store."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        conn = self._get_connection()
        if checkpoint_id:
            row = conn.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?
                """,
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_ns = ?
                ORDER BY checkpoint_id DESC
                LIMIT 1
                """,
                (thread_id, checkpoint_ns),
            ).fetchone()

        if row is None:
            return None

        c_id = row["checkpoint_id"]
        parent_id = row["parent_checkpoint_id"]
        c_type = row["type"]
        c_bytes = row["checkpoint"]
        m_type = row["metadata_type"]
        m_bytes = row["metadata"]

        raw_checkpoint: Checkpoint = self.serde.loads_typed((c_type, c_bytes))
        metadata: CheckpointMetadata = (
            self.serde.loads_typed((m_type, m_bytes)) if m_type and m_bytes else {}
        )

        # Load channel values from blobs table
        channel_values = self._load_blobs(
            thread_id,
            checkpoint_ns,
            raw_checkpoint.get("channel_versions", {}),
        )
        checkpoint: Checkpoint = {
            **raw_checkpoint,
            "channel_values": channel_values,
        }

        # Load pending writes
        writes = self._ordered_writes(thread_id, checkpoint_ns, c_id)
        pending_writes: list[PendingWrite] = [
            (task_id, channel, self.serde.loads_typed(v))
            for task_id, channel, v, _ in writes
        ]

        parent_config: RunnableConfig | None = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": c_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the filter criteria."""
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        conn = self._get_connection()
        query = [
            "SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata",
            "FROM checkpoints",
            "WHERE thread_id = ? AND checkpoint_ns = ?",
        ]
        params: list[Any] = [thread_id, checkpoint_ns]

        if before and (before_id := get_checkpoint_id(before)):
            query.append("AND checkpoint_id < ?")
            params.append(before_id)

        query.append("ORDER BY checkpoint_id DESC")
        if limit:
            query.append(f"LIMIT {int(limit)}")

        rows = conn.execute(" ".join(query), params).fetchall()
        for r in rows:
            c_id = r["checkpoint_id"]
            c_type = r["type"]
            c_bytes = r["checkpoint"]
            m_type = r["metadata_type"]
            m_bytes = r["metadata"]

            raw_checkpoint: Checkpoint = self.serde.loads_typed((c_type, c_bytes))
            metadata: CheckpointMetadata = (
                self.serde.loads_typed((m_type, m_bytes)) if m_type and m_bytes else {}
            )

            channel_values = self._load_blobs(
                thread_id,
                checkpoint_ns,
                raw_checkpoint.get("channel_versions", {}),
            )
            checkpoint: Checkpoint = {
                **raw_checkpoint,
                "channel_values": channel_values,
            }

            writes = self._ordered_writes(thread_id, checkpoint_ns, c_id)
            pending_writes: list[PendingWrite] = [
                (task_id, channel, self.serde.loads_typed(v))
                for task_id, channel, v, _ in writes
            ]

            parent_config: RunnableConfig | None = None
            if r["parent_checkpoint_id"]:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": r["parent_checkpoint_id"],
                    }
                }

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": c_id,
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the SQLite database."""
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")

        conn = self._get_connection()

        # 1. Save new channel blobs
        values: dict[str, Any] = c.pop("channel_values", {})
        for channel, version in new_versions.items():
            if channel in values:
                blob_type, blob_bytes = self.serde.dumps_typed(values[channel])
            else:
                blob_type, blob_bytes = "empty", b""

            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoint_blobs
                (thread_id, checkpoint_ns, channel, version, type, blob)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, checkpoint_ns, channel, str(version), blob_type, blob_bytes),
            )

        # 2. Serialize checkpoint and metadata
        c_type, c_bytes = self.serde.dumps_typed(c)
        full_metadata = get_checkpoint_metadata(config, metadata)
        m_type, m_bytes = self.serde.dumps_typed(full_metadata)

        conn.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (thread_id, checkpoint_ns, checkpoint_id, parent_id, c_type, c_bytes, m_type, m_bytes),
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes to the SQLite database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        conn = self._get_connection()
        for idx, (channel, value) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            blob_type, blob_bytes = self.serde.dumps_typed(value)

            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoint_writes
                (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    mapped_idx,
                    channel,
                    blob_type,
                    blob_bytes,
                    task_path,
                ),
            )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread."""
        conn = self._get_connection()
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)
