from router.executor import RouterExecutor
from router.health import HealthStore
from providers.base import ProviderRequest, ProviderResult
from providers.factory import PROVIDERS
from router.errors import RouterError, ErrorType


provider = PROVIDERS["ollama"]
original_chat = provider.chat

calls = []


def fake_chat(model, request):
    calls.append(model)

    request_id = request.metadata["request_id"]

    if model == "gemma4:31b-cloud":
        raise RouterError(
            ErrorType.PROVIDER_FAILURE,
            "Simulated tool-call provider failure",
            provider="ollama",
            model=model,
            retryable=True,
        )

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
                    "arguments": '{"city":"Bengaluru"}',
                },
            }
        ],
        finish_reason="tool_calls",
    )


provider.chat = fake_chat


try:
    health = HealthStore(
        provider_failure_cooldown=60,
    )

    executor = RouterExecutor(
        providers=PROVIDERS,
        health_store=health,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Bengaluru?",
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
        tool_choice={
            "type": "function",
            "function": {
                "name": "get_weather",
            },
        },
    )

    result = executor.route_and_execute(
        request,
        required_capabilities={"tools"},
        provider="ollama",
    )

    request_id = request.metadata["request_id"]

    print("=== TOOL-CALL FALLBACK TEST ===")
    print("request_id:", request_id)
    print("success:", result.success)
    print("provider:", result.provider)
    print("model:", result.model)
    print("content:", repr(result.content))
    print("finish_reason:", result.finish_reason)
    print("tool_calls:", result.tool_calls)
    print("calls:", calls)

    assert result.success
    assert result.finish_reason == "tool_calls"
    assert result.content == ""

    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]

    assert tool_call["id"] == f"call_{request_id}_0"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert (
        tool_call["function"]["arguments"]
        == '{"city":"Bengaluru"}'
    )

    assert calls == [
        "gemma4:31b-cloud",
        "nemotron-3-nano:30b-cloud",
    ]

    record = health.get(
        "ollama",
        "gemma4:31b-cloud",
    )

    assert record.state.value == "degraded"
    assert not record.available

    print()
    print("fallback model:", result.model)
    print("failed model health:", record.state)
    print("failed model available:", record.available)
    print("RESULT: PASS")

finally:
    provider.chat = original_chat
