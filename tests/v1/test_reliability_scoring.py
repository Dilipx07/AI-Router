from providers.base import ProviderUsage
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import (
    score_candidate,
    rank_candidates,
)


metrics = MetricsStore()


reliable = ModelCandidate(
    provider="ollama",
    model="reliable-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)


unreliable = ModelCandidate(
    provider="ollama",
    model="unreliable-model",
    priority=61,
    capabilities=frozenset({"chat"}),
)


# ------------------------------------------------------------
# Cold start
# ------------------------------------------------------------

cold = score_candidate(
    reliable,
    metrics_store=metrics,
)

print("=== SUCCESS-RATE SCORING ===")
print("cold score:", cold.score)
print("cold reasons:", cold.reasons)

assert cold.score == 600
assert "no reliability history" in cold.reasons


# ------------------------------------------------------------
# Reliable model: 4/4 success
# ------------------------------------------------------------

for _ in range(4):

    metrics.record_success(
        "ollama",
        "reliable-model",
        usage=ProviderUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        elapsed=1.0,
    )


# ------------------------------------------------------------
# Unreliable model: 2/4 success
# ------------------------------------------------------------

for _ in range(2):

    metrics.record_success(
        "ollama",
        "unreliable-model",
        usage=ProviderUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        elapsed=1.0,
    )


for _ in range(2):

    metrics.record_failure(
        "ollama",
        "unreliable-model",
        usage=ProviderUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        elapsed=1.0,
        error_type="timeout",
    )


reliable_scored = score_candidate(
    reliable,
    metrics_store=metrics,
)

unreliable_scored = score_candidate(
    unreliable,
    metrics_store=metrics,
)


print()
print("reliable score:", reliable_scored.score)
print("reliable reasons:", reliable_scored.reasons)

print("unreliable score:", unreliable_scored.score)
print("unreliable reasons:", unreliable_scored.reasons)


# reliable:
# 60 * 10 = 600
# latency = 1 second -> 25
# success = 100% -> 0
# final = 575

assert reliable_scored.score == 575


# unreliable:
# 61 * 10 = 610
# latency = 1 second -> 25
# success = 50% -> 100
# final = 485

assert unreliable_scored.score == 335


assert any(
    "reliability penalty" in reason
    for reason in unreliable_scored.reasons
)


# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

ranked = rank_candidates(
    [
        unreliable,
        reliable,
    ],
    metrics_store=metrics,
)

print()
print("ranked:", [
    (item.candidate.model, item.score)
    for item in ranked
])


assert ranked[0].candidate.model == "reliable-model"
assert ranked[1].candidate.model == "unreliable-model"


print()
print("cold-start reliability handling: PASS")
print("success-rate penalty: PASS")
print("reliability-aware ranking: PASS")
print()
print("RESULT: PASS")
