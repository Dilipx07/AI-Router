import time

from providers.base import ProviderUsage

from router.cost import ModelPricing
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.policy import get_policy
from router.quota import QuotaLimit, QuotaStore
from router.scorer import (
    score_candidate,
    rank_candidates,
    calculate_runtime_score,
)


print("=" * 70)
print("PHASE 12 — FULL INTELLIGENT ROUTING REGRESSION")
print("=" * 70)


# ============================================================
# CANDIDATES
# ============================================================

fast = ModelCandidate(
    provider="ollama",
    model="fast-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)

reliable = ModelCandidate(
    provider="ollama",
    model="reliable-model",
    priority=61,
    capabilities=frozenset({"chat"}),
)

expensive = ModelCandidate(
    provider="ollama",
    model="expensive-model",
    priority=62,
    capabilities=frozenset({"chat"}),
)

quota_low = ModelCandidate(
    provider="ollama",
    model="quota-low-model",
    priority=63,
    capabilities=frozenset({"chat"}),
)


# ============================================================
# STORES
# ============================================================

metrics = MetricsStore()
quota = QuotaStore()


# ============================================================
# FAST MODEL
# ============================================================

metrics.record_success(
    "ollama",
    "fast-model",
    usage=ProviderUsage(
        prompt_tokens=10_000,
        completion_tokens=2_000,
        total_tokens=12_000,
    ),
    elapsed=1.0,
)


# ============================================================
# RELIABLE MODEL
# ============================================================

for _ in range(4):

    metrics.record_success(
        "ollama",
        "reliable-model",
        usage=ProviderUsage(
            prompt_tokens=10_000,
            completion_tokens=2_000,
            total_tokens=12_000,
        ),
        elapsed=1.0,
    )


# ============================================================
# EXPENSIVE MODEL
# ============================================================

metrics.record_success(
    "ollama",
    "expensive-model",
    usage=ProviderUsage(
        prompt_tokens=100_000,
        completion_tokens=20_000,
        total_tokens=120_000,
    ),
    elapsed=4.0,
)


# ============================================================
# QUOTA-LOW MODEL
# ============================================================

quota.set_limit(
    "ollama",
    "quota-low-model",
    QuotaLimit(
        max_tokens=100,
    ),
)

quota.record(
    "ollama",
    "quota-low-model",
    usage=ProviderUsage(
        prompt_tokens=80,
        completion_tokens=0,
        total_tokens=80,
    ),
)


# ============================================================
# 1. COLD START
# ============================================================

cold = ModelCandidate(
    provider="ollama",
    model="cold-model",
    priority=50,
    capabilities=frozenset({"chat"}),
)

cold_score = score_candidate(
    cold,
    metrics_store=metrics,
    quota_store=quota,
)

assert cold_score.score == 500

assert cold_score.breakdown["latency_penalty"] == 0
assert cold_score.breakdown["reliability_penalty"] == 0
assert cold_score.breakdown["recent_failure_penalty"] == 0
assert cold_score.breakdown["cost_penalty"] == 0
assert cold_score.breakdown["quota_penalty"] == 0

print("[1] Cold-start handling: PASS")


# ============================================================
# 2. LATENCY
# ============================================================

fast_score = score_candidate(
    fast,
    metrics_store=metrics,
)

assert fast_score.breakdown["latency_penalty"] == 25

print("[2] Latency scoring: PASS")


# ============================================================
# 3. RELIABILITY
# ============================================================

for _ in range(2):

    metrics.record_failure(
        "ollama",
        "reliable-model",
        usage=ProviderUsage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
        elapsed=1.0,
        error_type="timeout",
    )


reliable_score = score_candidate(
    reliable,
    metrics_store=metrics,
)

assert (
    reliable_score.breakdown["reliability_penalty"]
    == 66
)

print("[3] Reliability scoring: PASS")


# ============================================================
# 4. RECENT FAILURE
# ============================================================

record = metrics.get(
    "ollama",
    "reliable-model",
)

assert record.last_failure_at is not None

runtime = calculate_runtime_score(
    reliable,
    metrics,
)

assert (
    runtime.recent_failure_penalty
    == 150
)

print("[4] Recent-failure scoring: PASS")


# ============================================================
# 5. COST
# ============================================================

pricing = {
    "ollama/expensive-model": ModelPricing(
        input_per_million=20.0,
        output_per_million=20.0,
    )
}

expensive_score = score_candidate(
    expensive,
    metrics_store=metrics,
    pricing=pricing,
)

assert (
    expensive_score.breakdown["cost_penalty"]
    == 240
)

print("[5] Cost scoring: PASS")


# ============================================================
# 6. QUOTA PRESSURE
# ============================================================

quota_runtime = calculate_runtime_score(
    quota_low,
    metrics,
    quota_store=quota,
)

assert quota_runtime.quota_penalty == 80

print("[6] Quota-pressure scoring: PASS")


# ============================================================
# 7. POLICY
# ============================================================

policy_runtime = calculate_runtime_score(
    reliable,
    metrics,
    policy=get_policy("reliable"),
)

assert (
    policy_runtime.reliability_penalty
    >= 2 * 1
)

print("[7] Routing policies: PASS")


# ============================================================
# 8. EXPLAINABILITY
# ============================================================

explained = score_candidate(
    expensive,
    metrics_store=metrics,
    pricing=pricing,
)

assert (
    explained.score
    == explained.breakdown["final_score"]
)

required_keys = {
    "base_score",
    "latency_penalty",
    "reliability_penalty",
    "recent_failure_penalty",
    "cost_penalty",
    "quota_penalty",
    "disabled_penalty",
    "missing_capability_penalty",
    "capability_bonus",
    "final_score",
}

assert required_keys.issubset(
    explained.breakdown.keys()
)

print("[8] Explainable scoring: PASS")


# ============================================================
# 9. RANKING
# ============================================================

ranked = rank_candidates(
    [
        fast,
        reliable,
        expensive,
    ],
    metrics_store=metrics,
    pricing=pricing,
)

assert len(ranked) == 3

assert all(
    ranked[index].score
    >= ranked[index + 1].score
    for index in range(
        len(ranked) - 1
    )
)

print("[9] Deterministic ranking: PASS")


# ============================================================
# 10. POLICY DIFFERENTIATION
# ============================================================

balanced = calculate_runtime_score(
    reliable,
    metrics,
    policy=get_policy("balanced"),
)

fast_policy = calculate_runtime_score(
    reliable,
    metrics,
    policy=get_policy("fast"),
)

reliable_policy = calculate_runtime_score(
    reliable,
    metrics,
    policy=get_policy("reliable"),
)

cheap_policy = calculate_runtime_score(
    expensive,
    metrics,
    pricing=pricing,
    policy=get_policy("cheap"),
)

assert (
    balanced.total_penalty
    != fast_policy.total_penalty
)

assert (
    balanced.total_penalty
    != reliable_policy.total_penalty
)

assert (
    balanced.total_penalty
    != cheap_policy.total_penalty
)

print("[10] Policy differentiation: PASS")


print()
print("=" * 70)
print("PHASE 12 — RESULT: PASS")
print("=" * 70)
