from fastapi.testclient import TestClient

import router_api


client = TestClient(
    router_api.app
)


print("=== REQUEST ID TEST ===")


# ============================================================
# GENERATED REQUEST ID
# ============================================================

response = client.post(
    "/v1/chat/completions",

    json={
        "model": "ai-router",

        "messages": [
            {
                "role": "user",
                "content":
                    "Reply with exactly ID_OK",
            }
        ],
    },
)


print(
    "status:",
    response.status_code,
)

generated_id = (
    response.headers.get(
        "X-Request-ID"
    )
)

body = response.json()

response_id = (
    body
    .get("x_router", {})
    .get("request_id")
)

print(
    "X-Request-ID:",
    generated_id,
)

print(
    "response request_id:",
    response_id,
)


assert response.status_code == 200

assert generated_id

assert generated_id == response_id

assert len(generated_id) == 32

assert all(
    character in "0123456789abcdef"
    for character in generated_id
)


print(
    "request ID generated: PASS"
)

print(
    "response request ID: PASS"
)

print(
    "X-Request-ID header: PASS"
)

print(
    "request ID format: PASS"
)


# ============================================================
# CLIENT-SUPPLIED REQUEST ID
# ============================================================

custom_id = (
    "custom-request-id-12345"
)


response = client.post(
    "/v1/chat/completions",

    headers={
        "X-Request-ID":
            custom_id,
    },

    json={
        "model": "ai-router",

        "messages": [
            {
                "role": "user",
                "content":
                    "Reply with exactly CUSTOM_OK",
            }
        ],
    },
)


custom_body = (
    response.json()
)

custom_response_id = (
    custom_body
    .get("x_router", {})
    .get("request_id")
)

custom_header_id = (
    response.headers.get(
        "X-Request-ID"
    )
)


print()
print(
    "custom header:",
    custom_header_id,
)

print(
    "custom response:",
    custom_response_id,
)


assert custom_header_id == custom_id

assert custom_response_id == custom_id


print(
    "client request ID preserved: PASS"
)


# ============================================================
# STREAMING REQUEST ID
# ============================================================

response = client.post(
    "/v1/chat/completions",

    headers={
        "X-Request-ID":
            "stream-request-123",
    },

    json={
        "model": "ai-router",

        "stream": True,

        "messages": [
            {
                "role": "user",
                "content":
                    "Reply with exactly STREAM_OK",
            }
        ],
    },
)


print()
print(
    "stream X-Request-ID:",
    response.headers.get(
        "X-Request-ID"
    ),
)


assert (
    response.headers.get(
        "X-Request-ID"
    )
    == "stream-request-123"
)


print(
    "stream request ID: PASS"
)


print()
print("RESULT: PASS")
