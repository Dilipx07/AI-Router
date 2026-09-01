import json

from fastapi.testclient import TestClient

import router_api
from providers.base import ProviderRequest, ProviderResult
from router.engine import RouterEngine, RoutingRequest
from router.executor import RouterExecutor, ExecutionError
from router.health import HealthStore
from router.models import ModelCandidate
from router.errors import ErrorType


# ============================================================
# TEST HELPERS
# ============================================================

TOOL = {
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


def assert_tool_call(
    tool_call,
    expected_id=None,
):
    assert tool_call["type"] == "function"

    if expected_id is not None:
        assert tool_call["id"] == expected_id

    function = tool_call["function"]

    assert function["name"] == "get_weather"

    arguments = function["arguments"]

    assert isinstance(arguments, str)

    assert json.loads(arguments) == {
        "city": "Bengaluru",
    }


# ============================================================
# 1. EXECUTOR TOOL CALL
# ============================================================

def test_executor_tool_call():

    provider_calls = []

    class FakeProvider:

        def chat(self, model, request):

            provider_calls.append(
                {
                    "model": model,
                    "request": request,
                }
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

    health = HealthStore()

    executor = RouterExecutor(
        providers={
            "ollama": FakeProvider(),
        },
        health_store=health,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "Weather?",
            }
        ],
        tools=[TOOL],
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

    assert result.success
    assert result.content == ""
    assert result.finish_reason == "tool_calls"
    assert result.has_tool_calls
    assert result.usable

    assert len(result.tool_calls) == 1

    assert_tool_call(
        result.tool_calls[0],
        expected_id=(
            f"call_{request.metadata['request_id']}_0"
        ),
    )

    assert len(provider_calls) == 1
    assert provider_calls[0]["request"].tools == [TOOL]

    print("[1] Executor tool call: PASS")


# ============================================================
# 2. CAPABILITY FILTER
# ============================================================

def test_capability_filter():

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="no-tools",
                    priority=10,
                    capabilities=frozenset({"chat"}),
                ),
                ModelCandidate(
                    provider="ollama",
                    model="with-tools",
                    priority=20,
                    capabilities=frozenset({
                        "chat",
                        "tools",
                    }),
                ),
            ]

    health = HealthStore()

    router = RouterEngine(
        model_registry=Registry(),
        health_store=health,
    )

    decision = router.route(
        RoutingRequest(
            required_capabilities=frozenset({
                "tools",
            }),
            provider="ollama",
        )
    )

    assert decision.success
    assert (
        decision.candidate.identifier
        == "ollama/with-tools"
    )

    assert decision.candidates_considered == 2
    assert decision.candidates_available == 2
    assert decision.candidates_compatible == 1

    assert any(
        item["candidate"] == "ollama/no-tools"
        and item["reason"] == "missing_capabilities"
        and "tools" in item["missing"]
        for item in decision.rejected
    )

    print("[2] Capability filtering: PASS")


# ============================================================
# 3. API NON-STREAMING
# ============================================================

def test_api_non_streaming():

    original_execute_request = (
        router_api.execute_request
    )

    def fake_execute_request(
        request,
        *,
        required_capabilities=None,
        provider=None,
        model=None,
        metadata=None,
    ):

        assert required_capabilities == {"tools"}
        assert provider == "ollama"
        assert request.tools == [TOOL]

        return ProviderResult.success_result(
            provider="ollama",
            model="test-tool-model",
            content="",
            tool_calls=[
                {
                    "id": "call_regression_0",
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

    router_api.execute_request = fake_execute_request

    try:

        client = TestClient(
            router_api.app
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ai-router",
                "provider": "ollama",
                "messages": [
                    {
                        "role": "user",
                        "content": "Weather?",
                    }
                ],
                "tools": [TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                    },
                },
                "stream": False,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["object"] == "chat.completion"
        assert data["model"] == "ai-router"

        choice = data["choices"][0]

        assert (
            choice["finish_reason"]
            == "tool_calls"
        )

        message = choice["message"]

        assert message["role"] == "assistant"
        assert message["content"] == ""

        assert_tool_call(
            message["tool_calls"][0],
            expected_id="call_regression_0",
        )

        print("[3] API non-streaming tool call: PASS")

    finally:
        router_api.execute_request = (
            original_execute_request
        )


# ============================================================
# 4. API STREAMING
# ============================================================

def test_api_streaming():

    original_execute_request = (
        router_api.execute_request
    )

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
            model="test-tool-model",
            content="",
            tool_calls=[
                {
                    "id": "call_stream_regression_0",
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

    router_api.execute_request = fake_execute_request

    try:

        client = TestClient(
            router_api.app
        )

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ai-router",
                "messages": [
                    {
                        "role": "user",
                        "content": "Weather?",
                    }
                ],
                "tools": [TOOL],
                "stream": True,
            },
        )

        assert response.status_code == 200

        assert response.headers[
            "content-type"
        ].startswith(
            "text/event-stream"
        )

        events = []

        for raw_event in response.text.split(
            "\n\n"
        ):

            raw_event = raw_event.strip()

            if not raw_event:
                continue

            assert raw_event.startswith("data:")

            payload = raw_event[
                len("data:"):
            ].strip()

            if payload == "[DONE]":
                events.append("[DONE]")
            else:
                events.append(
                    json.loads(payload)
                )

        assert events[-1] == "[DONE]"

        json_events = [
            event
            for event in events
            if event != "[DONE]"
        ]

        assert json_events

        completion_id = (
            json_events[0]["id"]
        )

        assert all(
            event["id"] == completion_id
            for event in json_events
        )

        tool_chunks = []

        for event in json_events:

            delta = event[
                "choices"
            ][0].get(
                "delta",
                {},
            )

            if delta.get("tool_calls"):
                tool_chunks.append(
                    delta["tool_calls"]
                )

        assert tool_chunks

        metadata = tool_chunks[0][0]

        assert (
            metadata["id"]
            == "call_stream_regression_0"
        )

        assert (
            metadata["function"]["name"]
            == "get_weather"
        )

        arguments = ""

        for group in tool_chunks:

            for item in group:

                function = item.get(
                    "function",
                    {},
                )

                arguments += (
                    function.get(
                        "arguments",
                        "",
                    )
                    or ""
                )

        assert arguments == (
            '{"city":"Bengaluru"}'
        )

        final_choice = (
            json_events[-1]["choices"][0]
        )

        assert (
            final_choice["finish_reason"]
            == "tool_calls"
        )

        print("[4] API streaming tool call: PASS")

    finally:
        router_api.execute_request = (
            original_execute_request
        )


# ============================================================
# 5. TOOL FALLBACK
# ============================================================

def test_tool_fallback():

    provider = router_api

    # Reuse the already validated executor
    # fallback behavior through a dedicated fake provider.

    class FakeProvider:

        def __init__(self):
            self.calls = []

        def chat(self, model, request):

            self.calls.append(model)

            request_id = request.metadata[
                "request_id"
            ]

            if model == "first":

                raise __import__(
                    "router.errors",
                    fromlist=["RouterError"],
                ).RouterError(
                    ErrorType.PROVIDER_FAILURE,
                    "Simulated failure",
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
                        "id": (
                            f"call_{request_id}_0"
                        ),
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

    fake_provider = FakeProvider()
    health = HealthStore()

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="first",
                    priority=20,
                    capabilities=frozenset({
                        "chat",
                        "tools",
                    }),
                ),
                ModelCandidate(
                    provider="ollama",
                    model="second",
                    priority=10,
                    capabilities=frozenset({
                        "chat",
                        "tools",
                    }),
                ),
            ]

    router = RouterEngine(
        model_registry=Registry(),
        health_store=health,
    )

    executor = RouterExecutor(
        providers={
            "ollama": fake_provider,
        },
        health_store=health,
        router=router,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "Weather?",
            }
        ],
        tools=[TOOL],
    )

    result = executor.route_and_execute(
        request,
        required_capabilities={"tools"},
        provider="ollama",
    )

    assert result.success
    assert result.model == "second"
    assert result.finish_reason == "tool_calls"

    assert fake_provider.calls == [
        "first",
        "second",
    ]

    failed_record = health.get(
        "ollama",
        "first",
    )

    assert failed_record.state.value == "degraded"
    assert not failed_record.available

    assert_tool_call(
        result.tool_calls[0],
        expected_id=(
            f"call_{request.metadata['request_id']}_0"
        ),
    )

    print("[5] Tool-call fallback: PASS")


# ============================================================
# 6. UNSUPPORTED TOOLS
# ============================================================

def test_unsupported_tools():

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="no-tools",
                    priority=10,
                    capabilities=frozenset({
                        "chat",
                    }),
                ),
            ]

    class Provider:

        def __init__(self):
            self.calls = 0

        def chat(self, model, request):
            self.calls += 1
            raise AssertionError(
                "Provider must not be called"
            )

    provider = Provider()

    health = HealthStore()

    router = RouterEngine(
        model_registry=Registry(),
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
                "content": "Use weather tool.",
            }
        ],
        tools=[TOOL],
    )

    try:

        executor.route_and_execute(
            request,
            required_capabilities={"tools"},
            provider="ollama",
        )

        raise AssertionError(
            "Expected ExecutionError"
        )

    except ExecutionError as error:

        assert (
            error.error_type
            == ErrorType.MODEL_UNAVAILABLE
        )

        assert provider.calls == 0

        rejected = error.metadata[
            "routing_rejected"
        ]

        assert len(rejected) == 1

        assert (
            rejected[0]["reason"]
            == "missing_capabilities"
        )

        assert "tools" in (
            rejected[0]["missing"]
        )

    print("[6] Unsupported tools: PASS")


# ============================================================
# RUN ALL
# ============================================================

print("=" * 70)
print("PHASE 9 — FULL TOOL REGRESSION")
print("=" * 70)

test_executor_tool_call()
test_capability_filter()
test_api_non_streaming()
test_api_streaming()
test_tool_fallback()
test_unsupported_tools()

print()
print("=" * 70)
print("PHASE 9 — RESULT: PASS")
print("=" * 70)
