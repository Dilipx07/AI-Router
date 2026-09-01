from router.executor import RouterExecutor, ExecutionError
from router.health import HealthStore
from providers.base import ProviderRequest
from providers.factory import PROVIDERS
from router.errors import RouterError, ErrorType


provider = PROVIDERS["ollama"]
original_chat = provider.chat

calls = []

failure_types = {
    "gemma4:31b-cloud": ErrorType.TIMEOUT,
    "nemotron-3-nano:30b-cloud": ErrorType.RATE_LIMIT,
    "nemotron-3-super:cloud": ErrorType.NETWORK_FAILURE,
    "gpt-oss:20b-cloud": ErrorType.TIMEOUT,
    "nemotron-3-ultra:cloud": ErrorType.RATE_LIMIT,
    "gpt-oss:120b-cloud": ErrorType.TIMEOUT,
    "minimax-m3:cloud": ErrorType.NETWORK_FAILURE,
}


def fake_chat(model, request):
    calls.append(model)

    raise RouterError(
        failure_types[model],
        f"Simulated {failure_types[model].value}",
        provider="ollama",
        model=model,
        retryable=True,
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

    try:
        executor.route_and_execute(
            request,
            provider="ollama",
        )

        print("FAIL: execution unexpectedly succeeded")

    except ExecutionError as error:

        print("=== MIXED FAILURES ===")
        print("error_type:", error.error_type.value)
        print("message:", error.message)
        print("retryable:", error.retryable)
        print("attempt_count:", len(error.metadata["attempts"]))
        print("calls:", calls)

finally:
    provider.chat = original_chat
