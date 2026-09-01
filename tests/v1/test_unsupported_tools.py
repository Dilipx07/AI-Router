from router.executor import RouterExecutor, ExecutionError
from router.engine import RouterEngine
from router.health import HealthStore
from router.models import ModelCandidate
from providers.base import ProviderRequest, ProviderResult
from router.errors import ErrorType


class ToollessRegistry:
    def all(self, enabled_only=True):
        return [
            ModelCandidate(
                provider="ollama",
                model="test-no-tools-1",
                priority=10,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="test-no-tools-2",
                priority=20,
                capabilities=frozenset({"chat"}),
            ),
        ]


class TestProvider:
    def __init__(self):
        self.calls = 0

    def chat(self, model, request):
        self.calls += 1

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="SHOULD_NOT_BE_CALLED",
        )


provider = TestProvider()


# ------------------------------------------------------------
# Use the real registered provider name "ollama", but replace
# the model registry with models that do not support tools.
# ------------------------------------------------------------

router = RouterEngine(
    model_registry=ToollessRegistry(),
    health_store=HealthStore(),
)

executor = RouterExecutor(
    providers={"ollama": provider},
    health_store=HealthStore(),
    router=router,
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Use the weather tool.",
        }
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ],
)

try:
    executor.route_and_execute(
        request,
        required_capabilities={"tools"},
        provider="ollama",
    )

    print("RESULT: FAIL")
    print("Execution unexpectedly succeeded")

    raise AssertionError(
        "Tool request unexpectedly executed"
    )

except ExecutionError as error:

    print("=== UNSUPPORTED TOOL TEST ===")
    print("error_type:", error.error_type.value)
    print("message:", error.message)
    print("retryable:", error.retryable)

    print("provider calls:", provider.calls)

    print()
    print("routing rejected:")

    for item in error.metadata.get(
        "routing_rejected",
        [],
    ):
        print(item)

    assert error.error_type == (
        ErrorType.MODEL_UNAVAILABLE
    )

    assert provider.calls == 0

    rejected = error.metadata.get(
        "routing_rejected",
        [],
    )

    assert len(rejected) == 2

    assert all(
        item["reason"] == "missing_capabilities"
        for item in rejected
    )

    assert all(
        "tools" in item["missing"]
        for item in rejected
    )

    print()
    print("provider call prevented: True")
    print("all candidates rejected for tools: True")
    print("RESULT: PASS")
