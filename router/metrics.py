"""
AI Router v1
Runtime usage and performance accounting.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from providers.base import ProviderUsage


# ============================================================
# MODEL METRICS
# ============================================================

@dataclass
class ModelMetrics:
    """
    Runtime statistics for one provider/model pair.
    """

    provider: str
    model: str

    requests: int = 0

    successes: int = 0

    failures: int = 0

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    total_latency: float = 0.0

    last_latency: float = 0.0

    last_success_at: Optional[float] = None

    last_failure_at: Optional[float] = None

    last_error_type: Optional[str] = None

    @property
    def average_latency(self) -> float:
        if self.requests <= 0:
            return 0.0

        return (
            self.total_latency
            / self.requests
        )

    @property
    def success_rate(self) -> float:
        if self.requests <= 0:
            return 0.0

        return (
            self.successes
            / self.requests
        )

    @property
    def failure_rate(self) -> float:
        if self.requests <= 0:
            return 0.0

        return (
            self.failures
            / self.requests
        )


# ============================================================
# METRICS STORE
# ============================================================

class MetricsStore:
    """
    Thread-safe in-memory runtime metrics store.

    Runtime state can optionally be persisted through an injected
    persistence object exposing save_metrics(record).
    """

    def __init__(
        self,
        *,
        persistence: Optional[Any] = None,
    ) -> None:

        self._lock = threading.RLock()

        self._records: dict[
            str,
            ModelMetrics,
        ] = {}

        self.persistence = persistence

    # ========================================================
    # PROVIDER SNAPSHOT
    # ========================================================

    def by_provider(
        self,
        provider: str,
    ) -> list[ModelMetrics]:
        """
        Return metrics for all models belonging to a provider.
        """

        with self._lock:

            return [
                record
                for record in self._records.values()
                if record.provider == provider
            ]

    # ========================================================
    # MODEL SNAPSHOT
    # ========================================================

    def by_model(
        self,
        provider: str,
        model: str,
    ) -> Optional[ModelMetrics]:
        """
        Return metrics for one exact provider/model pair.
        """

        with self._lock:

            return self._records.get(
                self._key(
                    provider,
                    model,
                )
            )

    # ========================================================
    # KEY
    # ========================================================

    @staticmethod
    def _key(
        provider: str,
        model: str,
    ) -> str:
        return f"{provider}/{model}"

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        provider: str,
        model: str,
    ) -> ModelMetrics:
        """
        Return the existing metrics record or create one.

        The operation is protected so concurrent callers cannot
        create conflicting records for the same provider/model.
        """

        with self._lock:

            key = self._key(
                provider,
                model,
            )

            if key not in self._records:

                self._records[key] = ModelMetrics(
                    provider=provider,
                    model=model,
                )

            return self._records[key]

    # ========================================================
    # SUCCESS
    # ========================================================

    def record_success(
        self,
        provider: str,
        model: str,
        *,
        usage: Optional[ProviderUsage] = None,
        elapsed: float = 0.0,
    ) -> ModelMetrics:
        """
        Record one successful request atomically.
        """

        with self._lock:

            record = self.get(
                provider,
                model,
            )

            record.requests += 1

            record.successes += 1

            record.last_latency = max(
                0.0,
                float(elapsed),
            )

            record.total_latency += (
                record.last_latency
            )

            record.last_success_at = (
                time.time()
            )

            record.last_error_type = None

            if usage is not None:

                normalized = (
                    usage.normalized()
                )

                record.prompt_tokens += (
                    normalized.prompt_tokens
                )

                record.completion_tokens += (
                    normalized.completion_tokens
                )

                record.total_tokens += (
                    normalized.total_tokens
                )

            if self.persistence is not None:

                self.persistence.save_metrics(
                    record
                )

            return record

    # ========================================================
    # FAILURE
    # ========================================================

    def record_failure(
        self,
        provider: str,
        model: str,
        *,
        usage: Optional[ProviderUsage] = None,
        elapsed: float = 0.0,
        error_type: Optional[str] = None,
    ) -> ModelMetrics:
        """
        Record one failed request atomically.
        """

        with self._lock:

            record = self.get(
                provider,
                model,
            )

            record.requests += 1

            record.failures += 1

            record.last_latency = max(
                0.0,
                float(elapsed),
            )

            record.total_latency += (
                record.last_latency
            )

            record.last_failure_at = (
                time.time()
            )

            record.last_error_type = (
                error_type
            )

            if usage is not None:

                normalized = (
                    usage.normalized()
                )

                record.prompt_tokens += (
                    normalized.prompt_tokens
                )

                record.completion_tokens += (
                    normalized.completion_tokens
                )

                record.total_tokens += (
                    normalized.total_tokens
                )

            if self.persistence is not None:

                self.persistence.save_metrics(
                    record
                )

            return record

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(self) -> list[ModelMetrics]:
        """
        Return a snapshot of all records.

        Copies are returned so callers cannot accidentally
        mutate the store's internal dictionary while iterating.
        """

        with self._lock:

            return list(
                self._records.values()
            )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Reset metrics atomically.

        provider=None:
            reset everything

        provider specified, model=None:
            reset all models for provider

        provider + model:
            reset one exact model
        """

        with self._lock:

            if provider is None:

                self._records.clear()

                return

            if model is None:

                prefix = f"{provider}/"

                keys = [
                    key
                    for key in self._records
                    if key.startswith(prefix)
                ]

                for key in keys:

                    self._records.pop(
                        key,
                        None,
                    )

                return

            self._records.pop(
                self._key(
                    provider,
                    model,
                ),
                None,
            )


# ============================================================
# GLOBAL METRICS STORE
# ============================================================

METRICS = MetricsStore()
