from providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

from router.engine import RouterEngine, RoutingRequest
from router.executor import RouterExecutor
from router.health import HealthStore
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.quota import QuotaLimit, QuotaStore


class Registry:

    def all(self, enabled_only=True):

        return [
            ModelCandidate(
                provider="ollama",
                model="quota-exhausted",
                priority=20,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="quota-available",
                priority=10,
                capabilities=frozenset({"chat"}),
            ),
        ]


class Provider:

    def __init__(self):
        self.calls = []

    def chat(self, model, request):

        self.calls.append(model)

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="QUOTA_OK",
            usage=ProviderUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


quota = QuotaStore()

quota.set_limit(
    "ollama",
    "quota-exhausted",
    QuotaLimit(
        max_tokens=10,
    ),
)

health = HealthStore()

metrics = MetricsStore()

router = RouterEngine(
    model_registry=Registry(),
    health_store=health,
    quota_store=quota,
)

provider = Provider()

executor = RouterExecutor(
    providers={
        "ollama": provider,
    },
    health_store=health,
    metrics_store=metrics,
    quota_store=quota,
    router=router,
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: QUOTA_OK",
        }
    ]
)


# ------------------------------------------------------------
# Exhaust the first model before routing.
# ------------------------------------------------------------

quota.record(
    "ollama",
    "quota-exhausted",
    usage=ProviderUsage(
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
    ),
)


# ------------------------------------------------------------
# Route.
# ------------------------------------------------------------

decision = router.route(
    RoutingRequest(
        provider="ollama",
    )
)

print("=== QUOTA-AWARE ROUTING TEST ===")
print("selected:", decision.candidate.identifier)
print("ranked:", [
    item.candidate.identifier
    for item in decision.ranked
])

print()
print("rejected:")

for item in decision.rejected:
    print(item)


assert decision.success

assert (
    decision.candidate.identifier
    == "ollama/quota-available"
)

assert all(
    item.candidate.identifier
    != "ollama/quota-exhausted"
    for item in decision.ranked
)

assert any(
    item["candidate"]
    == "ollama/quota-exhausted"
    and item["reason"]
    == "quota_unavailable"
    for item in decision.rejected
)


# ------------------------------------------------------------
# Execute selected candidate.
# ------------------------------------------------------------

result = executor.route_and_execute(
    request,
    provider="ollama",
)

assert result.success
assert result.model == "quota-available"
assert result.content == "QUOTA_OK"

assert provider.calls == [
    "quota-available",
]


state = quota.get(
    "ollama",
    "quota-available",
)

assert state.requests == 1
assert state.total_tokens == 15


print()
print("quota-exhausted candidate not executed: True")
print("quota-available candidate selected: True")
print("actual execution recorded against quota: True")
print()
print("RESULT: PASS")
