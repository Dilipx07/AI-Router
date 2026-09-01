from fastapi.testclient import TestClient

import router_api
from providers.base import ProviderResult


client = TestClient(router_api.app)

original_execute_request = router_api.execute_request


def fake_execute_request(
    request,
    *,
    required_capabilities=None,
    provider=None,
    model=None,
    metadata=None,
):
    print("provider restriction:", provider)
    print("required capabilities:", required_capabilities)
    print("received tools:", request.tools)
    print("received tool_choice:", request.tool_choice)

    return ProviderResult.success_result(
        provider="ollama",
        model="test-tool-model",
        content="",
        tool_calls=[
            {
                "id": "call_weather_001",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Bengaluru"}',
                },
            }
        ],
        finish_reason="tool_calls",
    )


router_api.execute_request = fake_execute_request


try:
    body = {
        "model": "ai-router",
        "provider": "ollama",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Bengaluru?",
            }
        ],
        "tools": [
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
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "get_weather",
            },
        },
        "stream": False,
    }

    response = client.post(
        "/v1/chat/completions",
        json=body,
    )

    print()
    print("=== API TOOL CALL TEST ===")
    print("status:", response.status_code)

    data = response.json()

    print("object:", data.get("object"))
    print("model:", data.get("model"))

    choice = data["choices"][0]
    message = choice["message"]

    print("finish_reason:", choice["finish_reason"])
    print("message role:", message.get("role"))
    print("content:", repr(message.get("content")))
    print("tool_calls:", message.get("tool_calls"))

    assert response.status_code == 200
    assert data["object"] == "chat.completion"
    assert data["model"] == "ai-router"

    assert choice["finish_reason"] == "tool_calls"

    assert message["role"] == "assistant"
    assert message["content"] == ""

    tool_calls = message.get("tool_calls")

    assert tool_calls
    assert len(tool_calls) == 1

    assert tool_calls[0]["id"] == "call_weather_001"
    assert tool_calls[0]["type"] == "function"

    assert (
        tool_calls[0]["function"]["name"]
        == "get_weather"
    )

    assert (
        tool_calls[0]["function"]["arguments"]
        == '{"city":"Bengaluru"}'
    )

    print()
    print("RESULT: PASS")

finally:
    router_api.execute_request = original_execute_request
