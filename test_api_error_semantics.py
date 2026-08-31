from fastapi.testclient import TestClient

import router_api

from providers.base import ProviderUsage
from router.errors import ErrorType
from router.executor import ExecutionError


client = TestClient(router_api.app)


def make_error(
    error_type: ErrorType,
    *,
    provider: str = "ollama",
    model: str = "test-model",
    retryable: bool = False,
) -> ExecutionError:

    return ExecutionError(
        error_type,
        f"Simulated {error_type.value}",
        provider=provider,
        model=model,
        retryable=retryable,
        usage=ProviderUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        metadata={
            "attempts": [
                {
                    "provider": provider,
                    "model": model,
                    "success": False,
                    "error_type": error_type.value,
                }
            ],
            "routing_rejected": [],
        },
    )


print("=== API ERROR SEMANTICS TEST ===")


# ============================================================
# Test cases
# ============================================================

cases = [
    (ErrorType.INVALID_REQUEST, 400),
    (ErrorType.TOOL_UNSUPPORTED, 400),
    (ErrorType.CONTEXT_TOO_LARGE, 400),
    (ErrorType.CONTENT_FILTER, 400),

    (ErrorType.AUTH_FAILURE, 401),

    (ErrorType.PAYMENT_REQUIRED, 402),

    (ErrorType.MODEL_NOT_FOUND, 404),

    (ErrorType.RATE_LIMIT, 429),
    (ErrorType.QUOTA_EXCEEDED, 429),

    (ErrorType.TIMEOUT, 504),

    (ErrorType.NETWORK_FAILURE, 502),
    (ErrorType.PROVIDER_FAILURE, 502),
    (ErrorType.EMPTY_RESPONSE, 502),

    (ErrorType.MODEL_UNAVAILABLE, 503),

    (ErrorType.UNKNOWN, 500),
]


# ============================================================
# Patch runtime executor
# ============================================================

original_execute = (
    router_api.RUNTIME.executor.route_and_execute
)


for error_type, expected_status in cases:

    def failing_execute(
        request,
        *,
        required_capabilities=None,
        provider=None,
        model=None,
        metadata=None,
        _error_type=error_type,
    ):

        raise make_error(
            _error_type
        )

    router_api.RUNTIME.executor.route_and_execute = (
        failing_execute
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ai-router",
            "messages": [
                {
                    "role": "user",
                    "content": "test",
                }
            ],
        },
    )

    print(
        f"{error_type.value:22} "
        f"→ HTTP {response.status_code}"
    )

    assert response.status_code == expected_status

    data = response.json()

    assert "error" in data

    assert (
        data["error"]["type"]
        == error_type.value
    )

    assert (
        data["error"]["code"]
        == error_type.value
    )

    assert (
        error_type.value
        in data["error"]["message"]
    )

    x_router = data.get(
        "x_router"
    )

    assert x_router is not None

    assert x_router["provider"] == "ollama"
    assert x_router["model"] == "test-model"

    assert (
        x_router["retryable"]
        is False
    )

    assert len(
        x_router["attempts"]
    ) == 1

    assert (
        x_router["usage"]
        == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
    )


# ============================================================
# Restore runtime
# ============================================================

router_api.RUNTIME.executor.route_and_execute = (
    original_execute
)


print()
print("HTTP status mapping: PASS")
print("error payload structure: PASS")
print("router diagnostics: PASS")
print("failure usage preservation: PASS")
print()
print("RESULT: PASS")
