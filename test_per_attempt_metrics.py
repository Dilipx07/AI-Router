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
from router.errors import RouterError, ErrorType
from router.quota import QuotaStore

metrics = MetricsStore()
health = HealthStore()
quota = QuotaStore()

class Registry:

    def all(self, enabled_only=True):

        return [
            ModelCandidate(
                provider="ollama",
                model="first",
                priority=20,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="second",
                priority=10,
                capabilities=frozenset({"chat"}),
            ),
        ]


class Provider:

    def __init__(self):
        self.calls = []

    def chat(self, model, request):

        self.calls.append(model)

        if model == "first":

            raise RouterError(
                ErrorType.TIMEOUT,
                "Simulated timeout",
                provider="ollama",
                model=model,
                retryable=True,
                usage=ProviderUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
            )

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="PER_ATTEMPT_OK",
            usage=ProviderUsage(
                prompt_tokens=20,
                completion_tokens=3,
                total_tokens=23,
            ),
        )


metrics = MetricsStore()
health = HealthStore()
provider = Provider()

router = RouterEngine(
    model_registry=Registry(),
    health_store=health,
    quota_store=quota,
    metrics_store=metrics,
)

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
            "content": "Reply with exactly: PER_ATTEMPT_OK",
        }
    ]
)

result = executor.route_and_execute(
    request,
    provider="ollama",
)

first = metrics.get(
    "ollama",
    "first",
)

second = metrics.get(
    "ollama",
    "second",
)

print("=== PER-ATTEMPT METRICS TEST ===")
print("result:", result.content)
print("calls:", provider.calls)

print()
print("FIRST")
print("requests:", first.requests)
print("successes:", first.successes)
print("failures:", first.failures)
print("prompt_tokens:", first.prompt_tokens)
print("completion_tokens:", first.completion_tokens)
print("total_tokens:", first.total_tokens)
print("last_error_type:", first.last_error_type)

print()
print("SECOND")
print("requests:", second.requests)
print("successes:", second.successes)
print("failures:", second.failures)
print("prompt_tokens:", second.prompt_tokens)
print("completion_tokens:", second.completion_tokens)
print("total_tokens:", second.total_tokens)
print("last_error_type:", second.last_error_type)


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

assert result.success
assert result.model == "second"
assert result.content == "PER_ATTEMPT_OK"


# ------------------------------------------------------------
# Provider call order
# ------------------------------------------------------------

assert provider.calls == [
    "first",
    "second",
]


# ------------------------------------------------------------
# First model metrics
# ------------------------------------------------------------

assert first.requests == 1
assert first.successes == 0
assert first.failures == 1

assert first.prompt_tokens == 10
assert first.completion_tokens == 2
assert first.total_tokens == 12

assert first.last_error_type == "timeout"


# ------------------------------------------------------------
# Second model metrics
# ------------------------------------------------------------

assert second.requests == 1
assert second.successes == 1
assert second.failures == 0

assert second.prompt_tokens == 20
assert second.completion_tokens == 3
assert second.total_tokens == 23

assert second.last_error_type is None


print()
print("first attempt recorded independently: True")
print("second attempt recorded independently: True")
print("fallback did not merge model metrics: True")
print("RESULT: PASS")
