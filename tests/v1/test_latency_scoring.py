from providers.base import ProviderUsage
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import (
    score_candidate,
    rank_candidates,
)


metrics = MetricsStore()


fast = ModelCandidate(
    provider="ollama",
    model="fast-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)

slow = ModelCandidate(
    provider="ollama",
    model="slow-model",
    priority=61,
    capabilities=frozenset({"chat"}),
)


# ------------------------------------------------------------
# No history
# ------------------------------------------------------------

no_history = score_candidate(
    fast,
    metrics_store=metrics,
)

print("=== LATENCY-AWARE SCORING ===")
print("no history score:", no_history.score)
print("reasons:", no_history.reasons)

assert no_history.score == 600
assert "no latency history" in no_history.reasons


# ------------------------------------------------------------
# Add latency history
# ------------------------------------------------------------

metrics.record_success(
    "ollama",
    "slow-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=4.0,
)


metrics.record_success(
    "ollama",
    "fast-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=1.0,
)


slow_scored = score_candidate(
    slow,
    metrics_store=metrics,
)

fast_scored = score_candidate(
    fast,
    metrics_store=metrics,
)

print()
print("slow score:", slow_scored.score)
print("slow reasons:", slow_scored.reasons)

print("fast score:", fast_scored.score)
print("fast reasons:", fast_scored.reasons)


assert slow_scored.score == (
    610 - 100
)

assert fast_scored.score == (
    600 - 25
)

assert any(
    "latency penalty" in reason
    for reason in slow_scored.reasons
)


# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

ranked = rank_candidates(
    [
        slow,
        fast,
    ],
    metrics_store=metrics,
)

print()
print("ranked:", [
    (item.candidate.model, item.score)
    for item in ranked
])


assert ranked[0].candidate.model == "fast-model"
assert ranked[1].candidate.model == "slow-model"


print()
print("cold-start handling: PASS")
print("latency penalty: PASS")
print("latency-aware ranking: PASS")
print()
print("RESULT: PASS")
