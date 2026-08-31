"""
AI Router v1
Runtime health tracking for providers and models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import ErrorType, RouterError


# ============================================================
# HEALTH STATES
# ============================================================

class HealthState(str, Enum):

    HEALTHY = "healthy"

    DEGRADED = "degraded"

    RATE_LIMITED = "rate_limited"

    QUOTA_EXHAUSTED = "quota_exhausted"

    UNAVAILABLE = "unavailable"

    AUTH_FAILED = "auth_failed"

    UNKNOWN = "unknown"


# ============================================================
# HEALTH RECORD
# ============================================================

@dataclass
class HealthRecord:

    provider: str
    model: Optional[str] = None

    state: HealthState = HealthState.UNKNOWN

    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    last_error: Optional[str] = None
    last_error_type: Optional[str] = None

    last_success_at: Optional[float] = None
    last_failure_at: Optional[float] = None

    cooldown_until: float = 0.0

    metadata: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return time.time() >= self.cooldown_until and (
            self.state
            not in {
                HealthState.AUTH_FAILED,
                HealthState.QUOTA_EXHAUSTED,
            }
        )

    @property
    def cooldown_remaining(self) -> float:

        remaining = self.cooldown_until - time.time()

        return max(0.0, remaining)


# ============================================================
# HEALTH STORE
# ============================================================

class HealthStore:
    """
    In-memory runtime health registry.

    Configuration remains immutable.
    Health changes only affect runtime routing decisions.
    """

    def __init__(
        self,
        default_cooldown: float = 30.0,
        rate_limit_cooldown: float = 60.0,
        provider_failure_cooldown: float = 15.0,
    ) -> None:

        self.default_cooldown = default_cooldown
        self.rate_limit_cooldown = rate_limit_cooldown
        self.provider_failure_cooldown = provider_failure_cooldown

        self._records: dict[str, HealthRecord] = {}

    # ========================================================
    # KEY
    # ========================================================

    @staticmethod
    def _key(
        provider: str,
        model: Optional[str] = None,
    ) -> str:

        if model:
            return f"{provider}/{model}"

        return provider

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> HealthRecord:

        key = self._key(provider, model)

        if key not in self._records:

            self._records[key] = HealthRecord(
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
        model: Optional[str] = None,
    ) -> HealthRecord:

        record = self.get(provider, model)

        record.total_requests += 1
        record.successful_requests += 1

        record.consecutive_failures = 0

        record.state = HealthState.HEALTHY

        record.last_success_at = time.time()

        record.last_error = None
        record.last_error_type = None

        record.cooldown_until = 0.0

        return record

    # ========================================================
    # FAILURE
    # ========================================================

    def record_failure(
        self,
        error: RouterError,
    ) -> HealthRecord:

        record = self.get(
            provider=error.provider or "unknown",
            model=error.model,
        )

        record.total_requests += 1
        record.failed_requests += 1

        record.consecutive_failures += 1

        record.last_failure_at = time.time()

        record.last_error = error.message

        record.last_error_type = (
            error.error_type.value
        )

        record.state = self._state_from_error(
            error.error_type
        )

        record.cooldown_until = (
            time.time()
            + self._cooldown_for_error(
                error.error_type
            )
        )

        return record

    # ========================================================
    # ERROR → STATE
    # ========================================================

    @staticmethod
    def _state_from_error(
        error_type: ErrorType,
    ) -> HealthState:

        mapping = {

            ErrorType.AUTH_FAILURE:
                HealthState.AUTH_FAILED,

            ErrorType.RATE_LIMIT:
                HealthState.RATE_LIMITED,

            ErrorType.QUOTA_EXCEEDED:
                HealthState.QUOTA_EXHAUSTED,

            ErrorType.PAYMENT_REQUIRED:
                HealthState.QUOTA_EXHAUSTED,

            ErrorType.MODEL_NOT_FOUND:
                HealthState.UNAVAILABLE,

            ErrorType.MODEL_UNAVAILABLE:
                HealthState.UNAVAILABLE,

            ErrorType.TIMEOUT:
                HealthState.DEGRADED,

            ErrorType.NETWORK_FAILURE:
                HealthState.DEGRADED,

            ErrorType.PROVIDER_FAILURE:
                HealthState.DEGRADED,

            ErrorType.EMPTY_RESPONSE:
                HealthState.DEGRADED,

        }

        return mapping.get(
            error_type,
            HealthState.UNKNOWN,
        )

    # ========================================================
    # COOLDOWN
    # ========================================================

    def _cooldown_for_error(
        self,
        error_type: ErrorType,
    ) -> float:

        if error_type == ErrorType.RATE_LIMIT:

            return self.rate_limit_cooldown

        if error_type in {
            ErrorType.PROVIDER_FAILURE,
            ErrorType.TIMEOUT,
            ErrorType.NETWORK_FAILURE,
            ErrorType.EMPTY_RESPONSE,
        }:

            return self.provider_failure_cooldown

        if error_type in {
            ErrorType.AUTH_FAILURE,
            ErrorType.PAYMENT_REQUIRED,
            ErrorType.QUOTA_EXCEEDED,
            ErrorType.MODEL_NOT_FOUND,
            ErrorType.MODEL_UNAVAILABLE,
        }:

            return self.default_cooldown

        return self.default_cooldown

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> bool:

        return self.get(
            provider,
            model,
        ).available

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:

        if provider is None:

            self._records.clear()

            return

        key = self._key(
            provider,
            model,
        )

        self._records.pop(
            key,
            None,
        )

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(self) -> list[HealthRecord]:

        return list(
            self._records.values()
        )


# ============================================================
# GLOBAL HEALTH STORE
# ============================================================

HEALTH = HealthStore()