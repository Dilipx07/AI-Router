from providers.base import ProviderUsage
from router.cost import ModelPricing
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.scorer import (
    COST_WEIGHT,
    score_candidate,
)


metrics = MetricsStore()


cheap = ModelCandidate(
    provider="ollama",
    model="cheap-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)

expensive = ModelCandidate(
    provider="ollama",
    model="expensive-model",
    priority=61,
    capabilities=frozenset({"chat"}),
)


pricing = {
    "ollama/cheap-model": ModelPricing(
        input_per_million=1.0,
        output_per_million=1.0,
    ),
    "ollama/expensive-model": ModelPricing(
        input_per_million=20.0,
        output_per_million=20.0,
    ),
}


# ------------------------------------------------------------
# No history
# ------------------------------------------------------------

cold = score_candidate(
    cheap,
    metrics_store=metrics,
    pricing=pricing,
)

print("=== COST-AWARE SCORING ===")
print("cold score:", cold.score)
print("cold reasons:", cold.reasons)

assert cold.score == 600
assert "no cost history" in cold.reasons


# ------------------------------------------------------------
# Cheap model
# ------------------------------------------------------------

metrics.record_success(
    "ollama",
    "cheap-model",
    usage=ProviderUsage(
        prompt_tokens=100_000,
        completion_tokens=20_000,
        total_tokens=120_000,
    ),
    elapsed=1.0,
)


# $0.10 input + $0.02 output = $0.12
# Average cost per request = $0.12
# penalty = 12

cheap_scored = score_candidate(
    cheap,
    metrics_store=metrics,
    pricing=pricing,
)


# ------------------------------------------------------------
# Expensive model
# ------------------------------------------------------------

metrics.record_success(
    "ollama",
    "expensive-model",
    usage=ProviderUsage(
        prompt_tokens=100_000,
        completion_tokens=20_000,
        total_tokens=120_000,
    ),
    elapsed=1.0,
)


# $2.00 input + $0.40 output = $2.40
# penalty = 240

expensive_scored = score_candidate(
    expensive,
    metrics_store=metrics,
    pricing=pricing,
)


print()
print("cheap score:", cheap_scored.score)
print("cheap reasons:", cheap_scored.reasons)

print(
    "expensive score:",
    expensive_scored.score,
)

print(
    "expensive reasons:",
    expensive_scored.reasons,
)


assert cheap_scored.score == (
    600 - 25 - 12
)

assert expensive_scored.score == (
    610 - 25 - 240
)

assert expensive_scored.score < (
    cheap_scored.score
)


assert any(
    "cost penalty" in reason
    for reason in expensive_scored.reasons
)


print()
print("cold-start cost handling: PASS")
print("cost penalty: PASS")
print("cost-aware scoring: PASS")
print()
print("RESULT: PASS")
