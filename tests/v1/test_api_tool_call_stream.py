import json

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
        "stream": True,
    }

    response = client.post(
        "/v1/chat/completions",
        json=body,
    )

    print()
    print("=== API STREAMING TOOL CALL TEST ===")
    print("status:", response.status_code)
    print("content-type:", response.headers.get("content-type"))
    print()
    print("RAW SSE:")
    print(response.text)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )

    raw_events = [
        event.strip()
        for event in response.text.split("\n\n")
        if event.strip()
    ]

    events = []

    for event in raw_events:
        assert event.startswith("data:")

        payload = event[len("data:"):].strip()

        if payload == "[DONE]":
            events.append("[DONE]")
            continue

        events.append(json.loads(payload))

    assert events[-1] == "[DONE]"

    json_events = [
        event
        for event in events
        if event != "[DONE]"
    ]

    assert len(json_events) >= 3

    completion_id = json_events[0]["id"]

    # --------------------------------------------------------
    # Every chunk must use the same completion ID.
    # --------------------------------------------------------

    assert all(
        event["id"] == completion_id
        for event in json_events
    )

    # --------------------------------------------------------
    # Every chunk must be a chat completion chunk.
    # --------------------------------------------------------

    assert all(
        event["object"] == "chat.completion.chunk"
        for event in json_events
    )

    # --------------------------------------------------------
    # Initial role chunk.
    # --------------------------------------------------------

    first_choice = json_events[0]["choices"][0]

    assert first_choice["delta"]["role"] == "assistant"
    assert first_choice["finish_reason"] is None

    # --------------------------------------------------------
    # Find tool-call chunks.
    # --------------------------------------------------------

    tool_call_chunks = []

    for event in json_events:
        choice = event["choices"][0]
        delta = choice.get("delta", {})

        if delta.get("tool_calls"):
            tool_call_chunks.append(
                delta["tool_calls"]
            )

    assert tool_call_chunks

    # --------------------------------------------------------
    # Tool-call metadata.
    # --------------------------------------------------------

    first_tool = tool_call_chunks[0][0]

    assert first_tool["index"] == 0
    assert first_tool["id"] == "call_weather_001"
    assert first_tool["type"] == "function"

    function = first_tool["function"]

    assert function["name"] == "get_weather"

    # --------------------------------------------------------
    # Tool arguments.
    # --------------------------------------------------------

    argument_fragments = []

    for tool_chunk in tool_call_chunks:
        for item in tool_chunk:
            function_data = item.get("function", {})

            arguments = function_data.get(
                "arguments"
            )

            if arguments:
                argument_fragments.append(arguments)

    arguments = "".join(argument_fragments)

    assert arguments == '{"city":"Bengaluru"}'

    # --------------------------------------------------------
    # Final finish reason.
    # --------------------------------------------------------

    final_choice = json_events[-1]["choices"][0]

    assert final_choice["delta"] == {}
    assert final_choice["finish_reason"] == "tool_calls"

    print()
    print("completion ID:", completion_id)
    print("tool call ID: call_weather_001")
    print("tool name: get_weather")
    print("arguments:", arguments)
    print("final finish reason:", final_choice["finish_reason"])
    print("DONE marker: present")
    print()
    print("RESULT: PASS")

finally:
    router_api.execute_request = original_execute_request
