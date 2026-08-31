import tempfile
from pathlib import Path

from providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

from router.cost import (
    ModelPricing,
    calculate_cost,
)

from router.engine import RouterEngine

from router.executor import RouterExecutor

from router.health import HealthStore

from router.metrics import MetricsStore

from router.models import ModelCandidate

from router.persistence import RuntimePersistence

from router.quota import (
    QuotaLimit,
    QuotaStore,
)


print("=" * 70)
print("PHASE 11 — FULL RUNTIME ACCOUNTING REGRESSION")
print("=" * 70)


# ============================================================
# 1. METRICS
# ============================================================

metrics = MetricsStore()

metrics.record_success(
    "ollama",
    "phase11-model",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    ),
    elapsed=1.5,
)

record = metrics.get(
    "ollama",
    "phase11-model",
)

assert record.requests == 1
assert record.successes == 1
assert record.failures == 0
assert record.total_tokens == 15

print("[1] Metrics accounting: PASS")


# ============================================================
# 2. COST
# ============================================================

cost = calculate_cost(
    ProviderUsage(
        prompt_tokens=100_000,
        completion_tokens=20_000,
        total_tokens=120_000,
    ),
    ModelPricing(
        input_per_million=1.0,
        output_per_million=2.0,
    ),
)

assert cost.input_cost == 0.1
assert cost.output_cost == 0.04
assert cost.total_cost == 0.14

print("[2] Cost calculation: PASS")


# ============================================================
# 3. QUOTA
# ============================================================

quota = QuotaStore()

quota.set_limit(
    "ollama",
    "phase11-quota",
    QuotaLimit(
        max_requests=2,
        max_tokens=50,
    ),
)

quota.record(
    "ollama",
    "phase11-quota",
    usage=ProviderUsage(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    ),
)

status = quota.check(
    "ollama",
    "phase11-quota",
)

assert status.allowed
assert status.remaining_requests == 1
assert status.remaining_tokens == 20

print("[3] Quota tracking: PASS")


# ============================================================
# 4. QUOTA ROUTING
# ============================================================

class Registry:

    def all(self, enabled_only=True):

        return [
            ModelCandidate(
                provider="ollama",
                model="quota-blocked",
                priority=20,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="quota-ok",
                priority=10,
                capabilities=frozenset({"chat"}),
            ),
        ]


quota2 = QuotaStore()

quota2.set_limit(
    "ollama",
    "quota-blocked",
    QuotaLimit(
        max_tokens=1,
    ),
)

quota2.record(
    "ollama",
    "quota-blocked",
    usage=ProviderUsage(
        prompt_tokens=1,
        completion_tokens=0,
        total_tokens=1,
    ),
)

health2 = HealthStore()

router2 = RouterEngine(
    model_registry=Registry(),
    health_store=health2,
    quota_store=quota2,
)

decision = router2.route(
    __import__(
        "router.engine",
        fromlist=["RoutingRequest"],
    ).RoutingRequest(
        provider="ollama",
    )
)

assert decision.candidate is not None
assert (
    decision.candidate.identifier
    == "ollama/quota-ok"
)

assert any(
    item["candidate"]
    == "ollama/quota-blocked"
    and item["reason"]
    == "quota_unavailable"
    for item in decision.rejected
)

print("[4] Quota-aware routing: PASS")


# ============================================================
# 5. EXECUTOR + METRICS + QUOTA
# ============================================================

class Provider:

    def chat(self, model, request):

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="PHASE11_OK",
            usage=ProviderUsage(
                prompt_tokens=12,
                completion_tokens=3,
                total_tokens=15,
            ),
        )


metrics3 = MetricsStore()
quota3 = QuotaStore()
health3 = HealthStore()

router3 = RouterEngine(
    model_registry=Registry(),
    health_store=health3,
    quota_store=quota3,
)

quota3.set_limit(
    "ollama",
    "quota-blocked",
    QuotaLimit(
        max_tokens=1,
    ),
)

quota3.record(
    "ollama",
    "quota-blocked",
    usage=ProviderUsage(
        prompt_tokens=1,
        completion_tokens=0,
        total_tokens=1,
    ),
)

executor = RouterExecutor(
    providers={
        "ollama": Provider(),
    },
    health_store=health3,
    metrics_store=metrics3,
    quota_store=quota3,
    router=router3,
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly PHASE11_OK",
        }
    ]
)

result = executor.route_and_execute(
    request,
    provider="ollama",
)

assert result.success
assert result.content == "PHASE11_OK"

metric = metrics3.get(
    "ollama",
    "quota-ok",
)

state = quota3.get(
    "ollama",
    "quota-ok",
)

assert metric.requests == 1
assert metric.successes == 1
assert metric.total_tokens == 15

assert state.requests == 1
assert state.total_tokens == 15

print("[5] Executor metrics + quota integration: PASS")


# ============================================================
# 6. PERSISTENCE
# ============================================================

with tempfile.TemporaryDirectory() as temp_dir:

    db_path = (
        Path(temp_dir)
        / "phase11.db"
    )

    persistence = RuntimePersistence(
        db_path=db_path
    )

    persistent_metrics = MetricsStore(
        persistence=persistence
    )

    persistent_quota = QuotaStore(
        persistence=persistence
    )

    persistent_metrics.record_success(
        "ollama",
        "persist-test",
        usage=ProviderUsage(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
        elapsed=2.0,
    )

    persistent_quota.record(
        "ollama",
        "persist-test",
        usage=ProviderUsage(
            prompt_tokens=30,
            completion_tokens=5,
            total_tokens=35,
        ),
    )

    restored = RuntimePersistence(
        db_path=db_path
    )

    restored_metrics = restored.load_metrics()
    restored_quota = restored.load_quota()

    assert len(restored_metrics) == 1
    assert len(restored_quota) == 1

    assert restored_metrics[0].total_tokens == 25
    assert restored_quota[0].total_tokens == 35

print("[6] Persistence + restart restoration: PASS")


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("PHASE 11 — RESULT: PASS")
print("=" * 70)
