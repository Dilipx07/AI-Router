"""
AI Router v1 - Model Scoring

Responsible only for ranking model candidates.

The scorer does NOT:
- make API calls
- inspect provider health
- perform fallback
- mutate registry state

It produces a deterministic score from candidate metadata,
request requirements, optional runtime signals, and routing
policy.
"""

from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Iterable, Optional

from providers.base import ProviderUsage

from .cost import (
    ModelPricing,
    calculate_cost,
)
from .metrics import MetricsStore
from .models import ModelCandidate
from .policy import (
    RoutingPolicy,
    get_policy,
)
from .quota import QuotaStore


# ============================================================
# SCORE WEIGHTS
# ============================================================

# Higher priority is the primary configuration signal.
PRIORITY_WEIGHT = 10

# Capability matching.
CAPABILITY_MATCH_SCORE = 100

# Hard scoring penalties.
DISABLED_PENALTY = 10_000
MISSING_CAPABILITY_PENALTY = 5_000

# Runtime penalties.
LATENCY_WEIGHT = 25
SUCCESS_RATE_WEIGHT = 200

RECENT_FAILURE_WINDOW = 300.0
RECENT_FAILURE_PENALTY = 150

COST_WEIGHT = 100
QUOTA_WEIGHT = 100


# ============================================================
# SCORE RESULT
# ============================================================

@dataclass(frozen=True)
class ScoredCandidate:
    """
    Fully scored model candidate.

    breakdown contains the individual score components.
    """

    candidate: ModelCandidate

    score: int

    reasons: tuple[str, ...]

    breakdown: dict[str, int]


# ============================================================
# RUNTIME SCORE
# ============================================================

@dataclass(frozen=True)
class RuntimeScore:
    """
    Runtime penalties applied to a candidate.

    All penalty values are already adjusted using the selected
    routing policy.
    """

    latency_penalty: int = 0

    reliability_penalty: int = 0

    recent_failure_penalty: int = 0

    cost_penalty: int = 0

    quota_penalty: int = 0

    latency_reason: str = ""

    reliability_reason: str = ""

    recent_failure_reason: str = ""

    cost_reason: str = ""

    quota_reason: str = ""

    @property
    def total_penalty(self) -> int:
        """
        Return the total runtime penalty.
        """

        return (
            self.latency_penalty
            + self.reliability_penalty
            + self.recent_failure_penalty
            + self.cost_penalty
            + self.quota_penalty
        )


# ============================================================
# QUOTA PENALTY
# ============================================================

def quota_penalty(
    candidate: ModelCandidate,
    quota_store: QuotaStore,
) -> tuple[int, str]:
    """
    Calculate a soft quota-pressure penalty.

    Exhausted candidates should already have been removed by
    the engine's hard quota filter.

    Candidates without a configured token quota receive no
    quota-pressure penalty.
    """

    status = quota_store.check(
        candidate.provider,
        candidate.model,
    )

    limit = quota_store.get_limit(
        candidate.provider,
        candidate.model,
    )

    if limit is None:

        return (
            0,
            "no quota limit",
        )

    if (
        limit.max_tokens is None
        or limit.max_tokens <= 0
    ):

        return (
            0,
            "no token quota limit",
        )

    if not status.allowed:

        return (
            0,
            "quota exhausted",
        )

    if status.remaining_tokens is None:

        return (
            0,
            "no token quota limit",
        )

    ratio = (
        status.remaining_tokens
        / limit.max_tokens
    )

    ratio = min(
        1.0,
        max(
            0.0,
            ratio,
        ),
    )

    pressure = 1.0 - ratio

    # round() avoids floating-point truncation such as:
    # 0.1 * 100 -> 9.999999... -> int(...) == 9
    penalty = round(
        pressure * QUOTA_WEIGHT
    )

    return (
        penalty,
        f"quota pressure penalty: {penalty} "
        f"({ratio:.1%} remaining)",
    )


# ============================================================
# COST PENALTY
# ============================================================

def cost_penalty(
    candidate: ModelCandidate,
    metrics_store: MetricsStore,
    pricing: Optional[
        dict[str, ModelPricing]
    ] = None,
) -> tuple[int, str]:
    """
    Calculate a soft cost penalty using average cost per request.

    Candidates without usage history or pricing receive no
    penalty.
    """

    if not pricing:

        return (
            0,
            "no cost pricing",
        )

    model_key = candidate.identifier

    model_pricing = pricing.get(
        model_key
    )

    if model_pricing is None:

        return (
            0,
            "no cost pricing",
        )

    record = metrics_store.by_model(
        candidate.provider,
        candidate.model,
    )

    if record is None:

        return (
            0,
            "no cost history",
        )

    if record.requests <= 0:

        return (
            0,
            "no cost history",
        )

    usage = ProviderUsage(
        prompt_tokens=record.prompt_tokens,

        completion_tokens=(
            record.completion_tokens
        ),

        total_tokens=record.total_tokens,
    )

    cost = calculate_cost(
        usage,
        model_pricing,
    )

    average_cost = (
        cost.total_cost
        / record.requests
    )

    penalty = int(
        average_cost * COST_WEIGHT
    )

    return (
        penalty,
        f"cost penalty: {penalty} "
        f"(avg {average_cost:.6f}/request)",
    )


# ============================================================
# LATENCY PENALTY
# ============================================================

def latency_penalty(
    candidate: ModelCandidate,
    metrics_store: MetricsStore,
) -> tuple[int, str]:
    """
    Calculate a latency-based score adjustment.

    Candidates without runtime history receive no penalty.
    Lower latency produces a smaller penalty.
    """

    record = metrics_store.by_model(
        candidate.provider,
        candidate.model,
    )

    if record is None:

        return (
            0,
            "no latency history",
        )

    if record.requests <= 0:

        return (
            0,
            "no latency history",
        )

    latency = max(
        0.0,
        record.average_latency,
    )

    penalty = int(
        latency * LATENCY_WEIGHT
    )

    return (
        penalty,
        f"latency penalty: {penalty} "
        f"({latency:.2f}s)",
    )


# ============================================================
# RELIABILITY PENALTY
# ============================================================

def reliability_penalty(
    candidate: ModelCandidate,
    metrics_store: MetricsStore,
) -> tuple[int, str]:
    """
    Calculate a reliability-based score adjustment.

    Candidates without runtime history receive no penalty.
    """

    record = metrics_store.by_model(
        candidate.provider,
        candidate.model,
    )

    if record is None:

        return (
            0,
            "no reliability history",
        )

    if record.requests <= 0:

        return (
            0,
            "no reliability history",
        )

    success_rate = min(
        1.0,
        max(
            0.0,
            record.success_rate,
        ),
    )

    penalty = int(
        (1.0 - success_rate)
        * SUCCESS_RATE_WEIGHT
    )

    return (
        penalty,
        f"reliability penalty: {penalty} "
        f"({success_rate:.2%} success)",
    )


# ============================================================
# RECENT FAILURE PENALTY
# ============================================================

def recent_failure_penalty(
    candidate: ModelCandidate,
    metrics_store: MetricsStore,
) -> tuple[int, str]:
    """
    Penalize candidates that have failed recently.

    Candidates without failure history receive no penalty.
    """

    record = metrics_store.by_model(
        candidate.provider,
        candidate.model,
    )

    if record is None:

        return (
            0,
            "no recent failure",
        )

    if record.last_failure_at is None:

        return (
            0,
            "no recent failure",
        )

    age = (
        time.time()
        - record.last_failure_at
    )

    if age < 0:
        age = 0.0

    if age > RECENT_FAILURE_WINDOW:

        return (
            0,
            f"no recent failure "
            f"({age:.1f}s ago)",
        )

    return (
        RECENT_FAILURE_PENALTY,
        f"recent failure penalty: "
        f"{RECENT_FAILURE_PENALTY} "
        f"({age:.1f}s ago)",
    )


# ============================================================
# RUNTIME SCORING
# ============================================================

def calculate_runtime_score(
    candidate: ModelCandidate,
    metrics_store: MetricsStore,
    pricing: Optional[
        dict[str, ModelPricing]
    ] = None,
    quota_store: Optional[
        QuotaStore
    ] = None,
    policy: RoutingPolicy | None = None,
) -> RuntimeScore:
    """
    Calculate the complete runtime score.

    Individual runtime penalties are calculated first and then
    adjusted using the selected routing policy.
    """

    policy = (
        policy
        if policy is not None
        else get_policy("balanced")
    )

    # --------------------------------------------------------
    # BASE PENALTIES
    # --------------------------------------------------------

    latency, latency_reason = latency_penalty(
        candidate,
        metrics_store,
    )

    reliability, reliability_reason = (
        reliability_penalty(
            candidate,
            metrics_store,
        )
    )

    recent_failure, recent_failure_reason = (
        recent_failure_penalty(
            candidate,
            metrics_store,
        )
    )

    cost, cost_reason = cost_penalty(
        candidate,
        metrics_store,
        pricing,
    )

    quota = 0

    quota_reason = "no quota data"

    if quota_store is not None:

        quota, quota_reason = quota_penalty(
            candidate,
            quota_store,
        )

    # --------------------------------------------------------
    # POLICY MULTIPLIERS
    # --------------------------------------------------------

    adjusted_latency = round(
        latency
        * policy.latency_multiplier
    )

    adjusted_reliability = round(
        reliability
        * policy.reliability_multiplier
    )

    adjusted_recent_failure = round(
        recent_failure
        * policy.recent_failure_multiplier
    )

    adjusted_cost = round(
        cost
        * policy.cost_multiplier
    )

    adjusted_quota = round(
        quota
        * policy.quota_multiplier
    )

    # --------------------------------------------------------
    # POLICY-AWARE REASONS
    # --------------------------------------------------------

    latency_reason = (
        latency_reason
        if adjusted_latency == latency
        else (
            f"{latency_reason}; "
            f"policy multiplier="
            f"{policy.latency_multiplier:.2f}; "
            f"adjusted={adjusted_latency}"
        )
    )

    reliability_reason = (
        reliability_reason
        if adjusted_reliability == reliability
        else (
            f"{reliability_reason}; "
            f"policy multiplier="
            f"{policy.reliability_multiplier:.2f}; "
            f"adjusted={adjusted_reliability}"
        )
    )

    recent_failure_reason = (
        recent_failure_reason
        if adjusted_recent_failure == recent_failure
        else (
            f"{recent_failure_reason}; "
            f"policy multiplier="
            f"{policy.recent_failure_multiplier:.2f}; "
            f"adjusted={adjusted_recent_failure}"
        )
    )

    cost_reason = (
        cost_reason
        if adjusted_cost == cost
        else (
            f"{cost_reason}; "
            f"policy multiplier="
            f"{policy.cost_multiplier:.2f}; "
            f"adjusted={adjusted_cost}"
        )
    )

    quota_reason = (
        quota_reason
        if adjusted_quota == quota
        else (
            f"{quota_reason}; "
            f"policy multiplier="
            f"{policy.quota_multiplier:.2f}; "
            f"adjusted={adjusted_quota}"
        )
    )

    return RuntimeScore(
        latency_penalty=adjusted_latency,

        reliability_penalty=(
            adjusted_reliability
        ),

        recent_failure_penalty=(
            adjusted_recent_failure
        ),

        cost_penalty=adjusted_cost,

        quota_penalty=adjusted_quota,

        latency_reason=latency_reason,

        reliability_reason=(
            reliability_reason
        ),

        recent_failure_reason=(
            recent_failure_reason
        ),

        cost_reason=cost_reason,

        quota_reason=quota_reason,
    )


# ============================================================
# SCORING
# ============================================================

def score_candidate(
    candidate: ModelCandidate,
    required_capabilities: Optional[
        Iterable[str]
    ] = None,
    *,
    metrics_store: Optional[
        MetricsStore
    ] = None,
    pricing: Optional[
        dict[str, ModelPricing]
    ] = None,
    quota_store: Optional[
        QuotaStore
    ] = None,
    policy: RoutingPolicy | None = None,
) -> ScoredCandidate:
    """
    Calculate a deterministic score for one model candidate.

    Higher score = better candidate.

    Required capabilities are treated as hard requirements.
    """

    required = set(
        required_capabilities or []
    )

    # --------------------------------------------------------
    # BASE SCORE
    # --------------------------------------------------------

    base_score = (
        candidate.priority
        * PRIORITY_WEIGHT
    )

    score = base_score

    reasons: list[str] = []

    # --------------------------------------------------------
    # BREAKDOWN COMPONENTS
    # --------------------------------------------------------

    latency_component = 0

    reliability_component = 0

    recent_failure_component = 0

    cost_component = 0

    quota_component = 0

    disabled_penalty = 0

    missing_capability_penalty = 0

    capability_bonus = 0

    # --------------------------------------------------------
    # RUNTIME SIGNALS
    # --------------------------------------------------------

    if metrics_store is not None:

        runtime = calculate_runtime_score(
            candidate,
            metrics_store,
            pricing,
            quota_store,
            policy,
        )

        latency_component = (
            runtime.latency_penalty
        )

        reliability_component = (
            runtime.reliability_penalty
        )

        recent_failure_component = (
            runtime.recent_failure_penalty
        )

        cost_component = (
            runtime.cost_penalty
        )

        quota_component = (
            runtime.quota_penalty
        )

        score -= runtime.total_penalty

        reasons.append(
            runtime.latency_reason
        )

        reasons.append(
            runtime.reliability_reason
        )

        reasons.append(
            runtime.recent_failure_reason
        )

        reasons.append(
            runtime.cost_reason
        )

        reasons.append(
            runtime.quota_reason
        )

    # --------------------------------------------------------
    # ENABLED
    # --------------------------------------------------------

    if candidate.enabled:

        reasons.append(
            "enabled"
        )

    else:

        disabled_penalty = (
            DISABLED_PENALTY
        )

        score -= disabled_penalty

        reasons.append(
            "disabled"
        )

    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    missing = (
        required
        - set(candidate.capabilities)
    )

    if missing:

        missing_capability_penalty = (
            MISSING_CAPABILITY_PENALTY
            * len(missing)
        )

        score -= (
            missing_capability_penalty
        )

        reasons.append(
            "missing capabilities: "
            + ", ".join(
                sorted(missing)
            )
        )

    else:

        if required:

            capability_bonus = (
                CAPABILITY_MATCH_SCORE
                * len(required)
            )

            score += capability_bonus

            reasons.append(
                "required capabilities matched"
            )

    # --------------------------------------------------------
    # STRUCTURED BREAKDOWN
    # --------------------------------------------------------

    breakdown = {
        "base_score": base_score,

        "latency_penalty":
            latency_component,

        "reliability_penalty":
            reliability_component,

        "recent_failure_penalty":
            recent_failure_component,

        "cost_penalty":
            cost_component,

        "quota_penalty":
            quota_component,

        "disabled_penalty":
            disabled_penalty,

        "missing_capability_penalty":
            missing_capability_penalty,

        "capability_bonus":
            capability_bonus,

        "final_score":
            score,
    }

    return ScoredCandidate(
        candidate=candidate,
        score=score,
        reasons=tuple(reasons),
        breakdown=breakdown,
    )


# ============================================================
# RANKING
# ============================================================

def rank_candidates(
    candidates: Iterable[ModelCandidate],
    required_capabilities: Optional[
        Iterable[str]
    ] = None,
    *,
    metrics_store: Optional[
        MetricsStore
    ] = None,
    pricing: Optional[
        dict[str, ModelPricing]
    ] = None,
    quota_store: Optional[
        QuotaStore
    ] = None,
    policy: RoutingPolicy | None = None,
) -> list[ScoredCandidate]:
    """
    Rank model candidates from best to worst.

    Tie-breaking is deterministic:

    1. Score
    2. Model priority
    3. Provider name
    4. Model name
    """

    scored = [
        score_candidate(
            candidate,
            required_capabilities=(
                required_capabilities
            ),
            metrics_store=metrics_store,
            pricing=pricing,
            quota_store=quota_store,
            policy=policy,
        )
        for candidate in candidates
    ]

    return sorted(
        scored,
        key=lambda item: (
            -item.score,
            -item.candidate.priority,
            item.candidate.provider,
            item.candidate.model,
        ),
    )


# ============================================================
# BEST CANDIDATE
# ============================================================

def select_best(
    candidates: Iterable[ModelCandidate],
    required_capabilities: Optional[
        Iterable[str]
    ] = None,
    *,
    metrics_store: Optional[
        MetricsStore
    ] = None,
    pricing: Optional[
        dict[str, ModelPricing]
    ] = None,
    quota_store: Optional[
        QuotaStore
    ] = None,
    policy: RoutingPolicy | None = None,
) -> Optional[ScoredCandidate]:
    """
    Return the highest-ranked candidate.

    Returns None when no candidates are supplied.
    """

    ranked = rank_candidates(
        candidates,
        required_capabilities=(
            required_capabilities
        ),
        metrics_store=metrics_store,
        pricing=pricing,
        quota_store=quota_store,
        policy=policy,
    )

    if not ranked:

        return None

    return ranked[0]
