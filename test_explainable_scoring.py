from providers.base import ProviderUsage

from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import score_candidate


metrics = MetricsStore()


candidate = ModelCandidate(
    provider="ollama",
    model="explain-model",
    priority=60,
    capabilities=frozenset({
        "chat",
    }),
)


metrics.record_success(
    "ollama",
    "explain-model",
    usage=ProviderUsage(
        prompt_tokens=100_000,
        completion_tokens=20_000,
        total_tokens=120_000,
    ),
    elapsed=1.0,
)


score = score_candidate(
    candidate,
    required_capabilities={"tools"},
    metrics_store=metrics,
)


print("=== EXPLAINABLE SCORE TEST ===")
print("score:", score.score)
print("reasons:", score.reasons)
print("breakdown:", score.breakdown)


assert score.breakdown["base_score"] == 600

assert score.breakdown["latency_penalty"] == 25

assert score.breakdown["reliability_penalty"] == 0

assert score.breakdown["recent_failure_penalty"] == 0

assert score.breakdown["cost_penalty"] == 0

assert score.breakdown["quota_penalty"] == 0

assert score.breakdown["capability_bonus"] == 0

assert score.breakdown["missing_capability_penalty"] == 5000

assert score.breakdown["disabled_penalty"] == 0

assert score.breakdown["final_score"] == (
    600 - 25 - 5000
)

assert score.score == (
    score.breakdown["final_score"]
)


# ------------------------------------------------------------
# Cold-start candidate
# ------------------------------------------------------------

cold_candidate = ModelCandidate(
    provider="ollama",
    model="cold-model",
    priority=50,
    capabilities=frozenset({
        "chat",
    }),
)


cold = score_candidate(
    cold_candidate
)


assert cold.breakdown["base_score"] == 500

assert cold.breakdown["latency_penalty"] == 0
assert cold.breakdown["reliability_penalty"] == 0
assert cold.breakdown["recent_failure_penalty"] == 0
assert cold.breakdown["cost_penalty"] == 0
assert cold.breakdown["quota_penalty"] == 0

assert cold.breakdown["capability_bonus"] == 0

assert cold.breakdown["disabled_penalty"] == 0

assert cold.breakdown["missing_capability_penalty"] == 0

assert cold.breakdown["final_score"] == 500


print()
print("runtime breakdown: PASS")
print("capability breakdown: PASS")
print("cold-start breakdown: PASS")
print("final-score consistency: PASS")
print()
print("RESULT: PASS")
