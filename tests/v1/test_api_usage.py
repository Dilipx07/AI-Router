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
        model="usage-test-model",
        content="USAGE_API_OK",
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
                    "content": "Reply with exactly: USAGE_API_OK",
                }
            ],
        },
    )

    print("=== API USAGE TEST ===")
    print("status:", response.status_code)

    data = response.json()

    print("content:", data["choices"][0]["message"]["content"])
    print("usage:", data["usage"])

    assert response.status_code == 200

    assert (
        data["choices"][0]["message"]["content"]
        == "USAGE_API_OK"
    )

    assert data["usage"] == {
        "prompt_tokens": 21,
        "completion_tokens": 4,
        "total_tokens": 25,
    }

    print()
    print("usage exposed correctly: True")
    print("RESULT: PASS")

finally:
    router_api.execute_request = original_execute_request
