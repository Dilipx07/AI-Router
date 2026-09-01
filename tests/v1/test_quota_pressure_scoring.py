from providers.base import ProviderUsage
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.quota import QuotaLimit, QuotaStore
from router.scorer import (
    QUOTA_WEIGHT,
    calculate_runtime_score,
    score_candidate,
)


quota = QuotaStore()

metrics = MetricsStore()


healthy_quota = ModelCandidate(
    provider="ollama",
    model="healthy-quota",
    priority=60,
    capabilities=frozenset({"chat"}),
)


low_quota = ModelCandidate(
    provider="ollama",
    model="low-quota",
    priority=61,
    capabilities=frozenset({"chat"}),
)


quota.set_limit(
    "ollama",
    "healthy-quota",
    QuotaLimit(
        max_tokens=100,
    ),
)

quota.set_limit(
    "ollama",
    "low-quota",
    QuotaLimit(
        max_tokens=100,
    ),
)


# healthy: 90 tokens remaining = 90%
quota.record(
    "ollama",
    "healthy-quota",
    usage=ProviderUsage(
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
    ),
)


# low: 20 tokens remaining = 20%
quota.record(
    "ollama",
    "low-quota",
    usage=ProviderUsage(
        prompt_tokens=40,
        completion_tokens=40,
        total_tokens=80,
    ),
)


# ------------------------------------------------------------
# Healthy quota
# ------------------------------------------------------------

healthy_runtime = calculate_runtime_score(
    healthy_quota,
    metrics,
    quota_store=quota,
)

print("=== QUOTA PRESSURE SCORING ===")

print(
    "healthy runtime:",
    healthy_runtime,
)

assert healthy_runtime.quota_penalty == 10


# ------------------------------------------------------------
# Low quota
# ------------------------------------------------------------

low_runtime = calculate_runtime_score(
    low_quota,
    metrics,
    quota_store=quota,
)

print()
print(
    "low-quota runtime:",
    low_runtime,
)

assert low_runtime.quota_penalty == 80

assert (
    low_runtime.quota_penalty
    > healthy_runtime.quota_penalty
)


# ------------------------------------------------------------
# Score integration
# ------------------------------------------------------------

healthy_score = score_candidate(
    healthy_quota,
    metrics_store=metrics,
    quota_store=quota,
)

low_score = score_candidate(
    low_quota,
    metrics_store=metrics,
    quota_store=quota,
)


print()
print("healthy score:", healthy_score.score)
print("low-quota score:", low_score.score)

print(
    "healthy reasons:",
    healthy_score.reasons,
)

print(
    "low-quota reasons:",
    low_score.reasons,
)


# healthy:
# 60 * 10 = 600
# quota penalty = 10
# score = 590

assert healthy_score.score == 590


# low:
# 61 * 10 = 610
# quota penalty = 80
# score = 530

assert low_score.score == 530

assert low_score.score < healthy_score.score

assert any(
    "quota pressure penalty" in reason
    for reason in low_score.reasons
)


# ------------------------------------------------------------
# No quota limit
# ------------------------------------------------------------

no_limit = ModelCandidate(
    provider="ollama",
    model="no-quota",
    priority=60,
    capabilities=frozenset({"chat"}),
)

runtime = calculate_runtime_score(
    no_limit,
    metrics,
    quota_store=quota,
)

assert runtime.quota_penalty == 0

assert runtime.quota_reason == (
    "no quota limit"
)


print()
print("quota-pressure calculation: PASS")
print("low-headroom penalty: PASS")
print("no-limit handling: PASS")
print("quota-aware scoring: PASS")
print()
print("RESULT: PASS")
