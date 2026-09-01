"""
AI Router v1
Core routing engine.

The engine is responsible for:

- discovering model candidates
- filtering unavailable candidates
- filtering request-size-incompatible candidates
- filtering incompatible capabilities
- ranking candidates
- producing a routing decision

The engine does NOT execute provider API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from config.registry import PROVIDER_REGISTRY

from .models import (
    MODEL_REGISTRY,
    ModelCandidate,
)
from .scorer import (
    ScoredCandidate,
    rank_candidates,
)


# ============================================================
# REQUEST TOKEN ESTIMATION
# ============================================================

# Conservative character/token ratio.

# This is deliberately conservative because underestimating
# a request can result in sending an oversized request to a
# constrained provider.
TOKEN_ESTIMATION_CHARS = 3.5


def estimate_text_tokens(
    value: Any,
) -> int:
    """
    Estimate the number of tokens represented by arbitrary text.

    This is a routing preflight estimate, not billing-grade
    tokenization.

    The conservative 3.5 chars/token ratio is intentionally used
    so request-size filtering prefers skipping a constrained model
    rather than accidentally sending an oversized request.
    """

    if value is None:
        return 0

    if isinstance(
        value,
        str,
    ):

        text = value

    else:

        text = str(value)

    if not text:

        return 0

    return max(
        1,
        int(
            len(text)
            / TOKEN_ESTIMATION_CHARS
        ),
    )


def estimate_request_tokens(
    request: RoutingRequest,
) -> int:
    """
    Estimate total request tokens.

    The estimate includes:

    - message content
    - message role
    - tool definitions
    - tool choice

    Request parameters such as max_tokens are intentionally not
    counted as input tokens.
    """

    total = 0

    for message in request.messages:

        total += estimate_text_tokens(
            message
        )

    if request.tools is not None:

        total += estimate_text_tokens(
            request.tools
        )

    if request.tool_choice is not None:

        total += estimate_text_tokens(
            request.tool_choice
        )

    return total


# ============================================================
# ROUTING REQUEST
# ============================================================

@dataclass(frozen=True)
class RoutingRequest:
    """
    Normalized requirements for one routing operation.

    `messages`, `tools`, and `tool_choice` are optional because
    older routing tests may construct RoutingRequest objects that
    only describe candidate-selection requirements.

    Production callers should populate them when available.
    """

    required_capabilities: frozenset[str] = (
        frozenset()
    )

    provider: Optional[str] = None

    model: Optional[str] = None

    metadata: Optional[
        dict[str, Any]
    ] = None

    # Optional request payload information used by request-size
    # preflight filtering.

    messages: tuple[
        dict[str, Any],
        ...
    ] = ()

    tools: Optional[Any] = None

    tool_choice: Optional[Any] = None

    def __post_init__(
        self,
    ) -> None:

        if self.metadata is None:

            object.__setattr__(
                self,
                "metadata",
                {},
            )

        if not isinstance(
            self.messages,
            tuple,
        ):

            object.__setattr__(
                self,
                "messages",
                tuple(
                    self.messages
                ),
            )


# ============================================================
# ROUTING DECISION
# ============================================================

@dataclass(frozen=True)
class RoutingDecision:
    """
    Result of the routing engine's selection phase.
    """

    selected: Optional[
        ScoredCandidate
    ]

    candidates_considered: int

    candidates_available: int

    candidates_compatible: int

    rejected: tuple[
        dict[str, Any],
        ...
    ] = ()

    ranked: tuple[
        ScoredCandidate,
        ...
    ] = ()

    @property
    def success(
        self,
    ) -> bool:

        return (
            self.selected
            is not None
        )

    @property
    def candidate(
        self,
    ) -> Optional[
        ModelCandidate
    ]:

        if not self.selected:

            return None

        return (
            self.selected.candidate
        )


# ============================================================
# ENGINE
# ============================================================

class RouterEngine:
    """
    Core deterministic routing engine.

    This class does not call external providers.
    """

    def __init__(
        self,
        *,
        model_registry=MODEL_REGISTRY,
        health_store=None,
        quota_store=None,
        metrics_store=None,
        policy=None,
    ) -> None:
        """
        Create a routing engine.

        Production runtime passes all dependencies explicitly.

        Defaults exist only for backwards compatibility with older
        isolated tests and do not reference mutable global stores.
        """

        from .health import HealthStore
        from .metrics import MetricsStore
        from .quota import QuotaStore
        from .policy import get_policy

        self.model_registry = (
            model_registry
        )

        self.health_store = (
            health_store
            if health_store is not None
            else HealthStore()
        )

        self.quota_store = (
            quota_store
            if quota_store is not None
            else QuotaStore()
        )

        self.metrics_store = (
            metrics_store
            if metrics_store is not None
            else MetricsStore()
        )

        self.policy = (
            policy
            if policy is not None
            else get_policy(
                "balanced"
            )
        )

    # ========================================================
    # CANDIDATE DISCOVERY
    # ========================================================

    def discover_candidates(
        self,
        request: RoutingRequest,
    ) -> list[ModelCandidate]:
        """
        Discover configured candidates matching explicit
        provider/model restrictions.
        """

        candidates = list(
            self.model_registry.all()
        )

        if request.provider:

            candidates = [
                candidate
                for candidate in candidates
                if candidate.provider
                == request.provider
            ]

        if request.model:

            candidates = [
                candidate
                for candidate in candidates
                if candidate.model
                == request.model
            ]

        return candidates

    # ========================================================
    # PROVIDER AVAILABILITY
    # ========================================================

    @staticmethod
    def provider_available(
        provider: str,
    ) -> bool:
        """
        Check whether provider configuration is usable.
        """

        config = (
            PROVIDER_REGISTRY.get(
                provider
            )
        )

        if config is None:

            return False

        return bool(
            config.available
        )

    # ========================================================
    # HEALTH FILTER
    # ========================================================

    def filter_health(
        self,
        candidates: Iterable[
            ModelCandidate
        ],
    ) -> tuple[
        list[ModelCandidate],
        list[dict[str, Any]],
    ]:

        available: list[
            ModelCandidate
        ] = []

        rejected: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:

            if not self.provider_available(
                candidate.provider
            ):

                rejected.append(
                    {
                        "candidate":
                            candidate.identifier,
                        "reason":
                            "provider_unavailable",
                    }
                )

                continue

            if not self.health_store.is_available(
                candidate.provider,
                candidate.model,
            ):

                record = (
                    self.health_store.get(
                        candidate.provider,
                        candidate.model,
                    )
                )

                rejected.append(
                    {
                        "candidate":
                            candidate.identifier,
                        "reason":
                            "health_unavailable",
                        "state":
                            record.state.value,
                        "cooldown_remaining":
                            record.cooldown_remaining,
                    }
                )

                continue

            available.append(
                candidate
            )

        return (
            available,
            rejected,
        )

    # ========================================================
    # QUOTA FILTER
    # ========================================================

    def filter_quota(
        self,
        candidates: Iterable[
            ModelCandidate
        ],
    ) -> tuple[
        list[ModelCandidate],
        list[dict[str, Any]],
    ]:

        available: list[
            ModelCandidate
        ] = []

        rejected: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:

            status = (
                self.quota_store.check(
                    candidate.provider,
                    candidate.model,
                )
            )

            if not status.allowed:

                rejected.append(
                    {
                        "candidate":
                            candidate.identifier,
                        "reason":
                            "quota_unavailable",
                        "quota_reason":
                            status.reason,
                        "remaining_requests":
                            status.remaining_requests,
                        "remaining_tokens":
                            status.remaining_tokens,
                    }
                )

                continue

            available.append(
                candidate
            )

        return (
            available,
            rejected,
        )

    # ========================================================
    # REQUEST SIZE FILTER
    # ========================================================

    @staticmethod
    def filter_request_size(
        candidates: Iterable[
            ModelCandidate
        ],
        request_tokens: int,
    ) -> tuple[
        list[ModelCandidate],
        list[dict[str, Any]],
    ]:
        """
        Reject candidates whose configured request-token limit is
        smaller than the estimated request size.

        A candidate with no configured limit remains eligible.

        This is a hard preflight filter and runs before ranking.
        """

        compatible: list[
            ModelCandidate
        ] = []

        rejected: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:

            limit = (
                candidate.max_request_tokens
            )

            if limit is None:

                compatible.append(
                    candidate
                )

                continue

            if request_tokens > limit:

                rejected.append(
                    {
                        "candidate":
                            candidate.identifier,

                        "reason":
                            "request_too_large",

                        "estimated_request_tokens":
                            request_tokens,

                        "max_request_tokens":
                            limit,

                        "excess_tokens":
                            request_tokens - limit,
                    }
                )

                continue

            compatible.append(
                candidate
            )

        return (
            compatible,
            rejected,
        )

    # ========================================================
    # CAPABILITY FILTER
    # ========================================================

    @staticmethod
    def filter_capabilities(
        candidates: Iterable[
            ModelCandidate
        ],
        required_capabilities: Iterable[
            str
        ],
    ) -> tuple[
        list[ModelCandidate],
        list[dict[str, Any]],
    ]:

        required = set(
            required_capabilities
        )

        compatible: list[
            ModelCandidate
        ] = []

        rejected: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:

            capabilities = set(
                candidate.capabilities
            )

            missing = (
                required
                - capabilities
            )

            if missing:

                rejected.append(
                    {
                        "candidate":
                            candidate.identifier,
                        "reason":
                            "missing_capabilities",
                        "missing":
                            sorted(missing),
                    }
                )

                continue

            compatible.append(
                candidate
            )

        return (
            compatible,
            rejected,
        )

    # ========================================================
    # ROUTE
    # ========================================================

    def route(
        self,
        request: RoutingRequest,
    ) -> RoutingDecision:

        candidates = (
            self.discover_candidates(
                request
            )
        )

        candidates_considered = len(
            candidates
        )

        if not candidates:

            return RoutingDecision(
                selected=None,
                candidates_considered=0,
                candidates_available=0,
                candidates_compatible=0,
            )

        # ----------------------------------------------------
        # HEALTH
        # ----------------------------------------------------

        available, health_rejected = (
            self.filter_health(
                candidates
            )
        )

        # ----------------------------------------------------
        # QUOTA
        # ----------------------------------------------------

        quota_available, quota_rejected = (
            self.filter_quota(
                available
            )
        )

        # ----------------------------------------------------
        # REQUEST SIZE
        # ----------------------------------------------------

        request_tokens = (
            estimate_request_tokens(
                request
            )
        )

        size_compatible, size_rejected = (
            self.filter_request_size(
                quota_available,
                request_tokens,
            )
        )

        # ----------------------------------------------------
        # CAPABILITIES
        # ----------------------------------------------------

        compatible, capability_rejected = (
            self.filter_capabilities(
                size_compatible,
                request.required_capabilities,
            )
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        ranked = rank_candidates(
            compatible,
            required_capabilities=(
                request.required_capabilities
            ),
            metrics_store=self.metrics_store,
            quota_store=self.quota_store,
            policy=self.policy,
        )

        selected = (
            ranked[0]
            if ranked
            else None
        )

        return RoutingDecision(
            selected=selected,
            candidates_considered=(
                candidates_considered
            ),
            candidates_available=len(
                available
            ),
            candidates_compatible=len(
                compatible
            ),
            rejected=tuple(
                health_rejected
                + quota_rejected
                + size_rejected
                + capability_rejected
            ),
            ranked=tuple(
                ranked
            ),
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def route_request(
    required_capabilities: Optional[
        Iterable[str]
    ] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    metadata: Optional[
        dict[str, Any]
    ] = None,
) -> RoutingDecision:
    """
    Backwards-compatible convenience wrapper.

    The actual production RouterEngine is owned by router.runtime.
    The import is intentionally deferred to avoid a circular import
    during runtime bootstrap.
    """

    request = RoutingRequest(
        required_capabilities=frozenset(
            required_capabilities
            or []
        ),
        provider=provider,
        model=model,
        metadata=metadata or {},
    )

    from .runtime import RUNTIME

    return RUNTIME.engine.route(
        request
    )
