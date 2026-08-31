from providers.base import ProviderRequest, ProviderResult, ProviderUsage
from router.engine import RouterEngine
from router.executor import RouterExecutor, ExecutionError
from router.health import HealthStore
from router.models import ModelCandidate
from router.errors import RouterError, ErrorType


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
                    prompt_tokens=21,
                    completion_tokens=4,
                    total_tokens=25,
                ),
            )

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="USAGE_FALLBACK_OK",
            usage=ProviderUsage(
                prompt_tokens=22,
                completion_tokens=8,
                total_tokens=30,
            ),
        )


health = HealthStore()

provider = Provider()

router = RouterEngine(
    model_registry=Registry(),
    health_store=health,
)

executor = RouterExecutor(
    providers={
        "ollama": provider,
    },
    health_store=health,
    router=router,
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: USAGE_FALLBACK_OK",
        }
    ]
)

result = executor.route_and_execute(
    request,
    provider="ollama",
)

print("=== FALLBACK USAGE ACCOUNTING ===")
print("success:", result.success)
print("model:", result.model)
print("content:", result.content)
print("result usage:", result.usage)
print("calls:", provider.calls)

assert result.success
assert result.model == "second"
assert result.content == "USAGE_FALLBACK_OK"

assert result.usage.prompt_tokens == 22
assert result.usage.completion_tokens == 8
assert result.usage.total_tokens == 30

assert provider.calls == [
    "first",
    "second",
]

print()
print("successful result usage preserved: True")
print("aggregate fallback usage is internal: PASS")


# ------------------------------------------------------------
# Now force every candidate to fail and inspect aggregate usage.
# ------------------------------------------------------------

class AllFailProvider:

    def chat(self, model, request):

        usage = {
            "first": ProviderUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
            "second": ProviderUsage(
                prompt_tokens=20,
                completion_tokens=3,
                total_tokens=23,
            ),
        }[model]

        raise RouterError(
            ErrorType.TIMEOUT,
            f"Simulated failure: {model}",
            provider="ollama",
            model=model,
            retryable=True,
            usage=usage,
        )


health2 = HealthStore()

router2 = RouterEngine(
    model_registry=Registry(),
    health_store=health2,
)

executor2 = RouterExecutor(
    providers={
        "ollama": AllFailProvider(),
    },
    health_store=health2,
    router=router2,
)

try:

    executor2.route_and_execute(
        request,
        provider="ollama",
    )

    raise AssertionError(
        "Expected ExecutionError"
    )

except ExecutionError as error:

    print()
    print("=== ALL FAILED USAGE ===")
    print("error_type:", error.error_type.value)

    aggregate = error.metadata["usage"]

    print("aggregate usage:", aggregate)

    assert aggregate.prompt_tokens == 30
    assert aggregate.completion_tokens == 5
    assert aggregate.total_tokens == 35

    assert len(
        error.metadata["attempts"]
    ) == 2

    print("aggregate prompt tokens: 30")
    print("aggregate completion tokens: 5")
    print("aggregate total tokens: 35")
    print()
    print("RESULT: PASS")
