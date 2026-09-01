from providers.base import ProviderUsage
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import (
    RuntimeScore,
    calculate_runtime_score,
    score_candidate,
)


metrics = MetricsStore()


candidate = ModelCandidate(
    provider="ollama",
    model="combined-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)


# ------------------------------------------------------------
# Cold start
# ------------------------------------------------------------

runtime = calculate_runtime_score(
    candidate,
    metrics,
)

print("=== COMBINED RUNTIME SCORE ===")
print("cold runtime:", runtime)

assert isinstance(
    runtime,
    RuntimeScore,
)

assert runtime.latency_penalty == 0
assert runtime.reliability_penalty == 0
assert runtime.total_penalty == 0

assert runtime.latency_reason == (
    "no latency history"
)

assert runtime.reliability_reason == (
    "no reliability history"
)


# ------------------------------------------------------------
# Runtime history
# ------------------------------------------------------------

metrics.record_success(
    "ollama",
    "combined-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=4.0,
)

metrics.record_failure(
    "ollama",
    "combined-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=4.0,
    error_type="timeout",
)


runtime = calculate_runtime_score(
    candidate,
    metrics,
)

print()
print("runtime:", runtime)

# 50% success → 100 reliability penalty
# 4 sec → 100 latency penalty

assert runtime.latency_penalty == 100
assert runtime.reliability_penalty == 100
assert runtime.total_penalty == 350


scored = score_candidate(
    candidate,
    metrics_store=metrics,
)

print()
print("score:", scored.score)
print("reasons:", scored.reasons)

# 60 * 10 = 600
# 600 - 200 = 400

assert scored.score == 250

assert (
    "latency penalty: 100 (4.00s)"
    in scored.reasons
)

assert (
    "reliability penalty: 100 (50.00% success)"
    in scored.reasons
)


print()
print("runtime components: PASS")
print("combined penalty: PASS")
print("score integration: PASS")
print("diagnostic reasons preserved: PASS")
print()
print("RESULT: PASS")
