from router.executor import RouterExecutor
from router.health import HealthStore
from providers.base import ProviderRequest, ProviderResult
from providers.factory import PROVIDERS


provider = PROVIDERS["ollama"]
original_chat = provider.chat

calls = []


def fake_chat(model, request):
    calls.append({
        "model": model,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
    })

    return ProviderResult.success_result(
        provider="ollama",
        model=model,
        content="",
        tool_calls=[
            {
                "id": "call_test_123",
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
    executor = RouterExecutor(
        providers=PROVIDERS,
        health_store=HealthStore(),
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

    print("=== TOOL CALL EXECUTOR TEST ===")
    print("success:", result.success)
    print("provider:", result.provider)
    print("model:", result.model)
    print("content:", repr(result.content))
    print("has_tool_calls:", result.has_tool_calls)
    print("tool_calls:", result.tool_calls)
    print("finish_reason:", result.finish_reason)
    print("usable:", result.usable)
    print("calls:", calls)

finally:
    provider.chat = original_chat
