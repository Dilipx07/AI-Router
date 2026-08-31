from router.executor import RouterExecutor
from router.engine import RouterEngine
from router.health import HealthStore
from router.models import MODEL_REGISTRY
from providers.base import ProviderRequest, ProviderResult
from providers.factory import PROVIDERS
from router.errors import RouterError, ErrorType

provider = PROVIDERS["ollama"]
original_chat = provider.chat

calls = []

def fake_chat(model, request):
    calls.append(model)

    if model == "gemma4:31b-cloud":
        raise RouterError(
            ErrorType.PROVIDER_FAILURE,
            "Simulated failure",
            provider="ollama",
            model=model,
            retryable=True,
        )

    return ProviderResult.success_result(
        provider="ollama",
        model=model,
        content="OK",
    )

provider.chat = fake_chat

try:
    health = HealthStore(
        provider_failure_cooldown=60
    )

    executor = RouterExecutor(
        providers=PROVIDERS,
        health_store=health,
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ]
    )

    print("=== FIRST REQUEST ===")

    result1 = executor.route_and_execute(request)

    print("success:", result1.success)
    print("model:", result1.model)
    print("calls:", calls)

    record = health.get(
        "ollama",
        "gemma4:31b-cloud",
    )

    print("gemma state:", record.state)
    print("gemma available:", record.available)
    print("cooldown:", round(record.cooldown_remaining, 2))

    calls.clear()

    print()
    print("=== SECOND REQUEST ===")

    result2 = executor.route_and_execute(request)

    print("success:", result2.success)
    print("model:", result2.model)
    print("calls:", calls)

finally:
    provider.chat = original_chat
