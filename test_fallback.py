from router.executor import RouterExecutor
from providers.base import ProviderRequest, ProviderResult
from providers.factory import PROVIDERS
from router.health import HealthStore
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
    executor = RouterExecutor(
        providers=PROVIDERS,
        health_store=HealthStore(),
    )

    request = ProviderRequest(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: OK",
            }
        ]
    )

    result = executor.route_and_execute(request)

    print("success:", result.success)
    print("provider:", result.provider)
    print("model:", result.model)
    print("content:", result.content)
    print("calls:", calls)

finally:
    provider.chat = original_chat
