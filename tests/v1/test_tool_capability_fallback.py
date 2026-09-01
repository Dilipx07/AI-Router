from router.executor import RouterExecutor
from router.engine import RouterEngine
from router.health import HealthStore
from router.models import ModelCandidate
from providers.base import ProviderRequest, ProviderResult


class MixedCapabilityRegistry:

    def all(self, enabled_only=True):
        return [
            ModelCandidate(
                provider="ollama",
                model="test-no-tools",
                priority=10,
                capabilities=frozenset({"chat"}),
            ),
            ModelCandidate(
                provider="ollama",
                model="test-tools",
                priority=20,
                capabilities=frozenset({
                    "chat",
                    "tools",
                }),
            ),
        ]


class TestProvider:

    def __init__(self):
        self.calls = []

    def chat(self, model, request):

        self.calls.append(model)

        if model == "test-no-tools":
            raise AssertionError(
                "Incompatible model was executed"
            )

        request_id = request.metadata["request_id"]

        return ProviderResult.success_result(
            provider="ollama",
            model=model,
            content="",
            tool_calls=[
                {
                    "id": f"call_{request_id}_0",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": (
                            '{"city":"Bengaluru"}'
                        ),
                    },
                }
            ],
            finish_reason="tool_calls",
        )


provider = TestProvider()
health = HealthStore()

router = RouterEngine(
    model_registry=MixedCapabilityRegistry(),
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

result = executor.route_and_execute(
    request,
    required_capabilities={"tools"},
    provider="ollama",
)

print("=== INCOMPATIBLE PRIMARY / TOOL FALLBACK TEST ===")
print("success:", result.success)
print("model:", result.model)
print("finish_reason:", result.finish_reason)
print("content:", repr(result.content))
print("tool_calls:", result.tool_calls)
print("calls:", provider.calls)

assert result.success
assert result.model == "test-tools"
assert result.finish_reason == "tool_calls"
assert result.content == ""

assert provider.calls == [
    "test-tools"
]

assert len(result.tool_calls) == 1

tool_call = result.tool_calls[0]

assert tool_call["type"] == "function"
assert tool_call["function"]["name"] == "get_weather"

assert (
    tool_call["function"]["arguments"]
    == '{"city":"Bengaluru"}'
)

rejected = router.route(
    __import__("router.engine", fromlist=["RoutingRequest"])
    .RoutingRequest(
        required_capabilities=frozenset({"tools"}),
        provider="ollama",
    )
).rejected

print()
print("routing rejected:")

for item in rejected:
    print(item)

assert any(
    item["candidate"] == "ollama/test-no-tools"
    and item["reason"] == "missing_capabilities"
    for item in rejected
)

assert all(
    item["candidate"] != "ollama/test-no-tools"
    for item in [
        {
            "candidate": call,
        }
        for call in provider.calls
    ]
)

print()
print("incompatible model executed: False")
print("tool-capable fallback executed: True")
print("RESULT: PASS")
