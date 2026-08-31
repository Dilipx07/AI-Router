import json

from fastapi.testclient import TestClient

import router_api

from providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

from router.engine import (
    RouterEngine,
    RoutingRequest,
)

from router.executor import (
    RouterExecutor,
    ExecutionError,
)

from router.health import HealthStore
from router.models import ModelCandidate
from router.errors import RouterError, ErrorType


# ============================================================
# TEST TOOL
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


# ============================================================
# 1. PROVIDER USAGE MODEL
# ============================================================

def test_usage_model():

    usage = ProviderUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=0,
    ).normalized()

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15
    assert usage.known

    print("[1] ProviderUsage model: PASS")


# ============================================================
# 2. OPENAI-COMPATIBLE EXTRACTION
# ============================================================

def test_openai_usage_extraction():

    from providers.openai_compatible import (
        OpenAICompatibleProvider,
    )

    usage = OpenAICompatibleProvider._extract_usage(
        {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 4,
                "total_tokens": 25,
            }
        }
    )

    assert usage.prompt_tokens == 21
    assert usage.completion_tokens == 4
    assert usage.total_tokens == 25

    print("[2] OpenAI-compatible usage extraction: PASS")


# ============================================================
# 3. GOOGLE EXTRACTION
# ============================================================

def test_google_usage_extraction():

    from providers.google import GoogleProvider

    usage = GoogleProvider._extract_usage(
        {
            "usageMetadata": {
                "promptTokenCount": 30,
                "candidatesTokenCount": 8,
                "totalTokenCount": 38,
            }
        }
    )

    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 38

    print("[3] Google usage extraction: PASS")


# ============================================================
# 4. CLOUDFLARE EXTRACTION
# ============================================================

def test_cloudflare_usage_extraction():

    from providers.cloudflare import CloudflareProvider

    usage = CloudflareProvider._extract_usage(
        {
            "result": {
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 8,
                }
            }
        }
    )

    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 38

    print("[4] Cloudflare usage extraction: PASS")


# ============================================================
# 5. EXECUTOR PRESERVATION
# ============================================================

def test_executor_preservation():

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="usage-test",
                    priority=10,
                    capabilities=frozenset({
                        "chat",
                    }),
                )
            ]

    class Provider:

        def chat(self, model, request):

            return ProviderResult.success_result(
                provider="ollama",
                model=model,
                content="USAGE_OK",
                usage=ProviderUsage(
                    prompt_tokens=21,
                    completion_tokens=4,
                    total_tokens=25,
                ),
            )

    health = HealthStore()

    router = RouterEngine(
        model_registry=Registry(),
        health_store=health,
    )

    executor = RouterExecutor(
        providers={
            "ollama": Provider(),
        },
        health_store=health,
        router=router,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "usage",
            }
        ]
    )

    result = executor.route_and_execute(
        request,
        provider="ollama",
    )

    assert result.success

    assert result.usage == ProviderUsage(
        prompt_tokens=21,
        completion_tokens=4,
        total_tokens=25,
    )

    print("[5] Executor usage preservation: PASS")


# ============================================================
# 6. FALLBACK AGGREGATE
# ============================================================

def test_fallback_aggregate():

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="first",
                    priority=20,
                    capabilities=frozenset({
                        "chat",
                    }),
                ),
                ModelCandidate(
                    provider="ollama",
                    model="second",
                    priority=10,
                    capabilities=frozenset({
                        "chat",
                    }),
                ),
            ]

    class Provider:

        def __init__(self):
            self.calls = []

        def chat(self, model, request):

            self.calls.append(model)

            if model == "first":

                raise RouterError(
                    ErrorType.TIMEOUT,
                    "Simulated failure",
                    provider="ollama",
                    model=model,
                    retryable=True,
                    usage=ProviderUsage(
                        prompt_tokens=10,
                        completion_tokens=2,
                        total_tokens=12,
                    ),
                )

            return ProviderResult.success_result(
                provider="ollama",
                model=model,
                content="FALLBACK_OK",
                usage=ProviderUsage(
                    prompt_tokens=20,
                    completion_tokens=3,
                    total_tokens=23,
                ),
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
                "content": "fallback",
            }
        ]
    )

    result = executor.route_and_execute(
        request,
        provider="ollama",
    )

    assert result.success
    assert result.model == "second"

    # Final response usage = successful attempt only.
    assert result.usage == ProviderUsage(
        prompt_tokens=20,
        completion_tokens=3,
        total_tokens=23,
    )

    assert provider.calls == [
        "first",
        "second",
    ]

    print("[6] Fallback final-result usage: PASS")


# ============================================================
# 7. ALL-FAILED AGGREGATE
# ============================================================

def test_all_failed_aggregate():

    class Registry:

        def all(self, enabled_only=True):

            return [
                ModelCandidate(
                    provider="ollama",
                    model="first",
                    priority=20,
                    capabilities=frozenset({
                        "chat",
                    }),
                ),
                ModelCandidate(
                    provider="ollama",
                    model="second",
                    priority=10,
                    capabilities=frozenset({
                        "chat",
                    }),
                ),
            ]

    class Provider:

        def chat(self, model, request):

            usage = {
                "first": ProviderUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
                "second": ProviderUsage(
                    prompt_tokens=20,
                    completion_tokens=3,
                    total_tokens=23,
                ),
            }[model]

            raise RouterError(
                ErrorType.TIMEOUT,
                "Simulated failure",
                provider="ollama",
                model=model,
                retryable=True,
                usage=usage,
            )

    health = HealthStore()

    router = RouterEngine(
        model_registry=Registry(),
        health_store=health,
    )

    executor = RouterExecutor(
        providers={
            "ollama": Provider(),
        },
        health_store=health,
        router=router,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "all fail",
            }
        ]
    )

    try:

        executor.route_and_execute(
            request,
            provider="ollama",
        )

        raise AssertionError(
            "Expected ExecutionError"
        )

    except ExecutionError as error:

        usage = error.metadata["usage"]

        assert usage == ProviderUsage(
            prompt_tokens=30,
            completion_tokens=5,
            total_tokens=35,
        )

    print("[7] All-failed aggregate usage: PASS")


# ============================================================
# 8. API NON-STREAMING
# ============================================================

def test_api_non_streaming():

    original = router_api.execute_request

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
            model="api-test",
            content="API_USAGE_OK",
            usage=ProviderUsage(
                prompt_tokens=21,
                completion_tokens=4,
                total_tokens=25,
            ),
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
                        "content": "usage",
                    }
                ],
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["usage"] == {
            "prompt_tokens": 21,
            "completion_tokens": 4,
            "total_tokens": 25,
        }

    finally:

        router_api.execute_request = original

    print("[8] API non-streaming usage: PASS")


# ============================================================
# 9. API STREAMING
# ============================================================

def test_api_streaming():

    original = router_api.execute_request

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
            model="api-stream-test",
            content="STREAM_USAGE_OK",
            usage=ProviderUsage(
                prompt_tokens=21,
                completion_tokens=4,
                total_tokens=25,
            ),
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
                        "content": "usage",
                    }
                ],
                "stream": True,
            },
        )

        assert response.status_code == 200

        events = []

        for raw_event in response.text.split("\n\n"):

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
                events.append(json.loads(payload))

        assert events[-1] == "[DONE]"

        json_events = [
            event
            for event in events
            if event != "[DONE]"
        ]

        assert json_events

        # No usage on earlier chunks.
        for event in json_events[:-1]:
            assert "usage" not in event

        final_event = json_events[-1]

        assert final_event["usage"] == {
            "prompt_tokens": 21,
            "completion_tokens": 4,
            "total_tokens": 25,
        }

    finally:

        router_api.execute_request = original

    print("[9] API streaming usage: PASS")


# ============================================================
# RUN
# ============================================================

print("=" * 70)
print("PHASE 10 — FULL USAGE REGRESSION")
print("=" * 70)

test_usage_model()
test_openai_usage_extraction()
test_google_usage_extraction()
test_cloudflare_usage_extraction()
test_executor_preservation()
test_fallback_aggregate()
test_all_failed_aggregate()
test_api_non_streaming()
test_api_streaming()

print()
print("=" * 70)
print("PHASE 10 — RESULT: PASS")
print("=" * 70)
