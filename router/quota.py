"""
AI Router v1
Quota tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from providers.base import ProviderUsage


# ============================================================
# QUOTA CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class QuotaLimit:
    """
    Usage limits for one provider/model.

    None means the dimension is unlimited/unknown.
    """

    max_requests: Optional[int] = None

    max_tokens: Optional[int] = None

    max_prompt_tokens: Optional[int] = None

    max_completion_tokens: Optional[int] = None


# ============================================================
# QUOTA STATE
# ============================================================

@dataclass
class QuotaState:

    provider: str
    model: str

    requests: int = 0

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    first_recorded_at: Optional[float] = None

    last_recorded_at: Optional[float] = None

    def record(
        self,
        usage: Optional[ProviderUsage] = None,
    ) -> None:

        now = time.time()

        if self.first_recorded_at is None:
            self.first_recorded_at = now

        self.last_recorded_at = now

        self.requests += 1

        if usage is None:
            return

        normalized = usage.normalized()

        self.prompt_tokens += (
            normalized.prompt_tokens
        )

        self.completion_tokens += (
            normalized.completion_tokens
        )

        self.total_tokens += (
            normalized.total_tokens
        )


# ============================================================
# QUOTA CHECK
# ============================================================

@dataclass(frozen=True)
class QuotaStatus:
    allowed: bool

    reason: Optional[str] = None

    remaining_requests: Optional[int] = None

    remaining_tokens: Optional[int] = None


def check_quota(
    state: QuotaState,
    limit: Optional[QuotaLimit],
) -> QuotaStatus:
    """
    Determine whether another request is permitted.
    """

    if limit is None:
        return QuotaStatus(
            allowed=True,
        )

    if (
        limit.max_requests is not None
        and state.requests >= limit.max_requests
    ):
        return QuotaStatus(
            allowed=False,
            reason="request_quota_exceeded",
            remaining_requests=0,
        )

    if (
        limit.max_tokens is not None
        and state.total_tokens >= limit.max_tokens
    ):
        return QuotaStatus(
            allowed=False,
            reason="token_quota_exceeded",
            remaining_tokens=0,
        )

    remaining_requests = None

    if limit.max_requests is not None:
        remaining_requests = max(
            0,
            limit.max_requests - state.requests,
        )

    remaining_tokens = None

    if limit.max_tokens is not None:
        remaining_tokens = max(
            0,
            limit.max_tokens - state.total_tokens,
        )

    return QuotaStatus(
        allowed=True,
        remaining_requests=remaining_requests,
        remaining_tokens=remaining_tokens,
    )


# ============================================================
# QUOTA STORE
# ============================================================

class QuotaStore:
    """
    Runtime quota state store.

    Quota tracking is observational at first. Routing integration
    is added in a later step.
    """

    def __init__(
        self,
        *,
        persistence=None,
    ) -> None:

        self._states: dict[str, QuotaState] = {}

        self._limits: dict[str, QuotaLimit] = {}

        self.persistence = persistence

    @staticmethod
    def _key(
        provider: str,
        model: str,
    ) -> str:
        return f"{provider}/{model}"

    # ========================================================
    # LIMITS
    # ========================================================

    def set_limit(
        self,
        provider: str,
        model: str,
        limit: QuotaLimit,
    ) -> None:

        self._limits[
            self._key(provider, model)
        ] = limit

    def get_limit(
        self,
        provider: str,
        model: str,
    ) -> Optional[QuotaLimit]:

        return self._limits.get(
            self._key(provider, model)
        )

    # ========================================================
    # STATE
    # ========================================================

    def get(
        self,
        provider: str,
        model: str,
    ) -> QuotaState:

        key = self._key(provider, model)

        if key not in self._states:

            self._states[key] = QuotaState(
                provider=provider,
                model=model,
            )

        return self._states[key]

    # ========================================================
    # RECORD
    # ========================================================

    def record(
        self,
        provider: str,
        model: str,
        *,
        usage: Optional[ProviderUsage] = None,
    ) -> QuotaState:

        state = self.get(
            provider,
            model,
        )

        state.record(
            usage
        )

        if self.persistence is not None:
            self.persistence.save_quota(
                state
            )
        return state

    # ========================================================
    # CHECK
    # ========================================================

    def check(
        self,
        provider: str,
        model: str,
    ) -> QuotaStatus:

        state = self.get(
            provider,
            model,
        )

        limit = self.get_limit(
            provider,
            model,
        )

        return check_quota(
            state,
            limit,
        )

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(self) -> list[QuotaState]:

        return list(
            self._states.values()
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:

        if provider is None:

            self._states.clear()

            return

        if model is None:

            prefix = f"{provider}/"

            keys = [
                key
                for key in self._states
                if key.startswith(prefix)
            ]

            for key in keys:
                self._states.pop(
                    key,
                    None,
                )

            return

        self._states.pop(
            self._key(
                provider,
                model,
            ),
            None,
        )

# ============================================================
# GLOBAL QUOTA STORE
# ============================================================

QUOTA = QuotaStore()