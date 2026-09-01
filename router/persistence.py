"""
AI Router v1
SQLite persistence for runtime metrics and quota state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import threading
from typing import Optional

from providers.base import ProviderUsage
from router.metrics import ModelMetrics
from router.quota import QuotaState


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "runtime.db"
)


class RuntimePersistence:
    """
    SQLite persistence layer.

    Connections are intentionally short-lived:
    one connection per operation.

    This means shutdown does not need to drain a connection pool,
    but we still expose an idempotent close() lifecycle method so
    the runtime has an explicit startup/shutdown contract.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
    ) -> None:

        self.db_path = Path(
            db_path
            if db_path is not None
            else DEFAULT_DB_PATH
        )

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self._closed = False

        self._initialize()

    # ========================================================
    # LIFECYCLE
    # ========================================================

    @property
    def closed(self) -> bool:
        """
        Return whether the persistence object has been closed.
        """

        with self._lock:
            return self._closed

    def startup(self) -> None:
        """
        Ensure the persistence layer is initialized.

        RuntimePersistence initializes eagerly in __init__, so
        startup() is intentionally idempotent.
        """

        with self._lock:

            if self._closed:

                raise RuntimeError(
                    "RuntimePersistence is already closed."
                )

            self._initialize()

    def close(self) -> None:
        """
        Close the persistence lifecycle.

        SQLite connections are not retained between operations,
        so there is no open connection to explicitly close here.
        """

        with self._lock:

            self._closed = True

    def shutdown(self) -> None:
        """
        Alias for close().
        """

        self.close()

    def _ensure_open(self) -> None:
        """
        Reject persistence operations after shutdown.
        """

        if self._closed:

            raise RuntimeError(
                "RuntimePersistence is closed."
            )

    # ========================================================
    # CONNECTION
    # ========================================================

    def _connect(self):
        self._ensure_open()

        return sqlite3.connect(
            self.db_path
        )

    # ========================================================
    # INITIALIZE
    # ========================================================

    def _initialize(self) -> None:

        conn = self._connect()

        try:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_metrics (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    total_latency REAL NOT NULL DEFAULT 0,
                    last_latency REAL NOT NULL DEFAULT 0,
                    last_success_at REAL,
                    last_failure_at REAL,
                    last_error_type TEXT,
                    PRIMARY KEY (provider, model)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quota_state (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    first_recorded_at REAL,
                    last_recorded_at REAL,
                    PRIMARY KEY (provider, model)
                )
                """
            )

            conn.commit()

        finally:

            conn.close()

    # ========================================================
    # METRICS SAVE
    # ========================================================

    def save_metrics(
        self,
        record: ModelMetrics,
    ) -> None:

        with self._lock:

            self._ensure_open()

            conn = self._connect()

            try:

                conn.execute(
                    """
                    INSERT INTO model_metrics (
                        provider,
                        model,
                        requests,
                        successes,
                        failures,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        total_latency,
                        last_latency,
                        last_success_at,
                        last_failure_at,
                        last_error_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model)
                    DO UPDATE SET
                        requests=excluded.requests,
                        successes=excluded.successes,
                        failures=excluded.failures,
                        prompt_tokens=excluded.prompt_tokens,
                        completion_tokens=excluded.completion_tokens,
                        total_tokens=excluded.total_tokens,
                        total_latency=excluded.total_latency,
                        last_latency=excluded.last_latency,
                        last_success_at=excluded.last_success_at,
                        last_failure_at=excluded.last_failure_at,
                        last_error_type=excluded.last_error_type
                    """,
                    (
                        record.provider,
                        record.model,
                        record.requests,
                        record.successes,
                        record.failures,
                        record.prompt_tokens,
                        record.completion_tokens,
                        record.total_tokens,
                        record.total_latency,
                        record.last_latency,
                        record.last_success_at,
                        record.last_failure_at,
                        record.last_error_type,
                    ),
                )

                conn.commit()

            finally:

                conn.close()

    # ========================================================
    # METRICS LOAD
    # ========================================================

    def load_metrics(
        self,
    ) -> list[ModelMetrics]:

        with self._lock:

            self._ensure_open()

            conn = self._connect()

            try:

                rows = conn.execute(
                    """
                    SELECT
                        provider,
                        model,
                        requests,
                        successes,
                        failures,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        total_latency,
                        last_latency,
                        last_success_at,
                        last_failure_at,
                        last_error_type
                    FROM model_metrics
                    """
                ).fetchall()

            finally:

                conn.close()

        return [
            ModelMetrics(
                provider=row[0],
                model=row[1],
                requests=row[2],
                successes=row[3],
                failures=row[4],
                prompt_tokens=row[5],
                completion_tokens=row[6],
                total_tokens=row[7],
                total_latency=row[8],
                last_latency=row[9],
                last_success_at=row[10],
                last_failure_at=row[11],
                last_error_type=row[12],
            )
            for row in rows
        ]

    # ========================================================
    # QUOTA SAVE
    # ========================================================

    def save_quota(
        self,
        state: QuotaState,
    ) -> None:

        with self._lock:

            self._ensure_open()

            conn = self._connect()

            try:

                conn.execute(
                    """
                    INSERT INTO quota_state (
                        provider,
                        model,
                        requests,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        first_recorded_at,
                        last_recorded_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, model)
                    DO UPDATE SET
                        requests=excluded.requests,
                        prompt_tokens=excluded.prompt_tokens,
                        completion_tokens=excluded.completion_tokens,
                        total_tokens=excluded.total_tokens,
                        first_recorded_at=excluded.first_recorded_at,
                        last_recorded_at=excluded.last_recorded_at
                    """,
                    (
                        state.provider,
                        state.model,
                        state.requests,
                        state.prompt_tokens,
                        state.completion_tokens,
                        state.total_tokens,
                        state.first_recorded_at,
                        state.last_recorded_at,
                    ),
                )

                conn.commit()

            finally:

                conn.close()

    # ========================================================
    # QUOTA LOAD
    # ========================================================

    def load_quota(
        self,
    ) -> list[QuotaState]:

        with self._lock:

            self._ensure_open()

            conn = self._connect()

            try:

                rows = conn.execute(
                    """
                    SELECT
                        provider,
                        model,
                        requests,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        first_recorded_at,
                        last_recorded_at
                    FROM quota_state
                    """
                ).fetchall()

            finally:

                conn.close()

        return [
            QuotaState(
                provider=row[0],
                model=row[1],
                requests=row[2],
                prompt_tokens=row[3],
                completion_tokens=row[4],
                total_tokens=row[5],
                first_recorded_at=row[6],
                last_recorded_at=row[7],
            )
            for row in rows
        ]


# ============================================================
# GLOBAL PERSISTENCE STORE
# ============================================================

PERSISTENCE = RuntimePersistence()
