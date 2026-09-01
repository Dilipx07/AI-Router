import time

from providers.base import ProviderUsage
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import (
    RECENT_FAILURE_PENALTY,
    calculate_runtime_score,
    score_candidate,
)


metrics = MetricsStore()


candidate = ModelCandidate(
    provider="ollama",
    model="failure-test",
    priority=60,
    capabilities=frozenset({"chat"}),
)


# ------------------------------------------------------------
# No failure history
# ------------------------------------------------------------

runtime = calculate_runtime_score(
    candidate,
    metrics,
)

print("=== FAILURE-AWARE SCORING ===")
print("no failure:", runtime)

assert runtime.recent_failure_penalty == 0
assert runtime.recent_failure_reason == (
    "no recent failure"
)


# ------------------------------------------------------------
# Recent failure
# ------------------------------------------------------------

metrics.record_failure(
    "ollama",
    "failure-test",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
    ),
    elapsed=1.0,
    error_type="timeout",
)


runtime = calculate_runtime_score(
    candidate,
    metrics,
)

print()
print("recent failure:", runtime)

assert (
    runtime.recent_failure_penalty
    == RECENT_FAILURE_PENALTY
)

assert "recent failure penalty" in (
    runtime.recent_failure_reason
)


scored = score_candidate(
    candidate,
    metrics_store=metrics,
)

print()
print("score:", scored.score)
print("reasons:", scored.reasons)

# Base:
# 60 * 10 = 600
#
# Failure metrics also produce:
# latency penalty = 25
# reliability penalty = 200
# recent failure = 150
#
# final = 225

assert scored.score == 225


# ------------------------------------------------------------
# Simulate old failure
# ------------------------------------------------------------

record = metrics.get(
    "ollama",
    "failure-test",
)

record.last_failure_at = (
    time.time() - 1000
)


runtime = calculate_runtime_score(
    candidate,
    metrics,
)

print()
print("old failure:", runtime)

assert runtime.recent_failure_penalty == 0

assert (
    "no recent failure"
    in runtime.recent_failure_reason
)


print()
print("no-history handling: PASS")
print("recent failure penalty: PASS")
print("expired failure penalty: PASS")
print("failure-aware scoring: PASS")
print()
print("RESULT: PASS")
