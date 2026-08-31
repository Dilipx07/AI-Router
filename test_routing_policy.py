from providers.base import ProviderUsage

from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.policy import get_policy
from router.scorer import calculate_runtime_score


metrics = MetricsStore()


candidate = ModelCandidate(
    provider="ollama",
    model="policy-model",
    priority=60,
    capabilities=frozenset({"chat"}),
)


metrics.record_success(
    "ollama",
    "policy-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=4.0,
)


metrics.record_failure(
    "ollama",
    "policy-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=4.0,
    error_type="timeout",
)


balanced = calculate_runtime_score(
    candidate,
    metrics,
    policy=get_policy("balanced"),
)

fast = calculate_runtime_score(
    candidate,
    metrics,
    policy=get_policy("fast"),
)

reliable = calculate_runtime_score(
    candidate,
    metrics,
    policy=get_policy("reliable"),
)

cheap = calculate_runtime_score(
    candidate,
    metrics,
    policy=get_policy("cheap"),
)


print("=== ROUTING POLICY TEST ===")

print("balanced:", balanced)
print("fast:", fast)
print("reliable:", reliable)
print("cheap:", cheap)


# Base penalties:
#
# latency = 4s * 25 = 100
# reliability = 50% failure → 100
# recent failure = 150
#
# Balanced = 350


assert balanced.total_penalty == 350


# Fast:
# latency: 100 * 2 = 200
# reliability: 100 * .75 = 75
# recent: 150 * .75 = 112.5 → 112
#
# total = 387


assert fast.total_penalty == 387


# Reliable:
# latency: 100 * .75 = 75
# reliability: 100 * 2 = 200
# recent: 150 * 2 = 300
#
# total = 575


assert reliable.total_penalty == 575


# Cheap:
# latency: 100 * .5 = 50
# reliability: 100 * .75 = 75
# recent: 150 * .75 = 112
# cost/quota = 0
#
# total = 237


assert cheap.total_penalty == 237


assert fast.total_penalty != balanced.total_penalty
assert reliable.total_penalty != balanced.total_penalty
assert cheap.total_penalty != balanced.total_penalty


# Unknown policy falls back to balanced.

unknown = get_policy(
    "does-not-exist"
)

assert unknown.name == "balanced"


print()
print("balanced policy: PASS")
print("fast policy: PASS")
print("reliable policy: PASS")
print("cheap policy: PASS")
print("unknown policy fallback: PASS")
print()
print("RESULT: PASS")
