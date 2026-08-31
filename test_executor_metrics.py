from providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

from router.engine import RouterEngine
from router.executor import RouterExecutor
from router.health import HealthStore
from router.metrics import MetricsStore
from router.models import ModelCandidate
from router.quota import QuotaStore


class Registry:

    def all(self, enabled_only=True):

        return [
            ModelCandidate(
                provider="ollama",
                model="metrics-test",
                priority=10,
                capabilities=frozenset({
                    "chat",
                }),
            )
        ]


class Provider:

    def chat(self, model, request):

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="METRICS_OK",
            usage=ProviderUsage(
                prompt_tokens=21,
                completion_tokens=4,
                total_tokens=25,
            ),
        )


metrics = MetricsStore()
health = HealthStore()
quota = QuotaStore()


router = RouterEngine(
    model_registry=Registry(),
    health_store=health,
    quota_store=quota,
    metrics_store=metrics,
)


executor = RouterExecutor(
    providers={
        "ollama": Provider(),
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
            "content": "Reply with exactly: METRICS_OK",
        }
    ]
)


result = executor.route_and_execute(
    request,
    provider="ollama",
)


record = metrics.get(
    "ollama",
    "metrics-test",
)


print("=== EXECUTOR METRICS TEST ===")
print("result:", result.content)
print("requests:", record.requests)
print("successes:", record.successes)
print("failures:", record.failures)
print("prompt_tokens:", record.prompt_tokens)
print("completion_tokens:", record.completion_tokens)
print("total_tokens:", record.total_tokens)
print("last_latency:", record.last_latency)


assert result.success
assert result.content == "METRICS_OK"

assert record.requests == 1
assert record.successes == 1
assert record.failures == 0

assert record.prompt_tokens == 21
assert record.completion_tokens == 4
assert record.total_tokens == 25

assert record.last_latency >= 0.0


print()
print("metrics recorded by executor: True")
print("RESULT: PASS")
