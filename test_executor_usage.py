from providers.base import ProviderRequest, ProviderResult, ProviderUsage
from router.engine import RouterEngine
from router.executor import RouterExecutor
from router.health import HealthStore
from router.models import ModelCandidate


class Registry:

    def all(self, enabled_only=True):
        return [
            ModelCandidate(
                provider="ollama",
                model="usage-test-model",
                priority=10,
                capabilities=frozenset({"chat"}),
            )
        ]


class FakeProvider:

    def chat(self, model, request):

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="USAGE_OK",
            usage=ProviderUsage(
                prompt_tokens=21,
                completion_tokens=4,
                total_tokens=25,
            ),
        )


health = HealthStore()

router = RouterEngine(
    model_registry=Registry(),
    health_store=health,
)

executor = RouterExecutor(
    providers={
        "ollama": FakeProvider(),
    },
    health_store=health,
    router=router,
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: USAGE_OK",
        }
    ]
)

result = executor.route_and_execute(
    request,
    provider="ollama",
)

print("=== EXECUTOR USAGE PRESERVATION ===")
print("success:", result.success)
print("content:", result.content)
print("provider:", result.provider)
print("model:", result.model)
print("usage:", result.usage)
print("prompt_tokens:", result.usage.prompt_tokens)
print("completion_tokens:", result.usage.completion_tokens)
print("total_tokens:", result.usage.total_tokens)

assert result.success
assert result.content == "USAGE_OK"

assert result.usage.prompt_tokens == 21
assert result.usage.completion_tokens == 4
assert result.usage.total_tokens == 25

assert result.usage.known

print()
print("usage preserved: True")
print("RESULT: PASS")
