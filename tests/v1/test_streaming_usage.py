import json

from fastapi.testclient import TestClient

import router_api
from providers.base import ProviderResult, ProviderUsage


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
    return ProviderResult.success_result(
        provider="ollama",
        model="usage-stream-test",
        content="STREAM_USAGE_OK",
        usage=ProviderUsage(
            prompt_tokens=21,
            completion_tokens=4,
            total_tokens=25,
        ),
    )


router_api.execute_request = fake_execute_request


try:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ai-router",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: STREAM_USAGE_OK",
                }
            ],
            "stream": True,
        },
    )

    print("=== STREAMING USAGE TEST ===")
    print("status:", response.status_code)
    print("content-type:", response.headers.get("content-type"))
    print()

    assert response.status_code == 200

    events = []

    for raw_event in response.text.split("\n\n"):

        raw_event = raw_event.strip()

        if not raw_event:
            continue

        assert raw_event.startswith("data:")

        payload = raw_event[len("data:"):].strip()

        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))

    assert events[-1] == "[DONE]"

    json_events = [
        event
        for event in events
        if event != "[DONE]"
    ]

    assert json_events

    final_event = json_events[-1]

    print("final event:", final_event)

    usage = final_event.get("usage")

    assert usage == {
        "prompt_tokens": 21,
        "completion_tokens": 4,
        "total_tokens": 25,
    }

    assert (
        final_event["choices"][0]["finish_reason"]
        == "stop"
    )

    # Usage should be present only on the final chunk.
    for event in json_events[:-1]:
        assert "usage" not in event

    print()
    print("usage present on final chunk: True")
    print("usage absent from earlier chunks: True")
    print("prompt_tokens:", usage["prompt_tokens"])
    print("completion_tokens:", usage["completion_tokens"])
    print("total_tokens:", usage["total_tokens"])
    print("RESULT: PASS")

finally:
    router_api.execute_request = original_execute_request
