from router.executor import RouterExecutor, ExecutionError
from router.health import HealthStore
from providers.base import ProviderRequest
from providers.factory import PROVIDERS
from router.errors import RouterError, ErrorType


provider = PROVIDERS["ollama"]
original_chat = provider.chat

calls = []


def fake_chat(model, request):
    calls.append(model)

    raise RouterError(
        ErrorType.TIMEOUT,
        "Simulated timeout",
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

        print("=== ALL CANDIDATES FAILED ===")
        print("error_type:", error.error_type.value)
        print("message:", error.message)
        print("provider:", error.provider)
        print("model:", error.model)
        print("retryable:", error.retryable)

        print()
        print("attempt_count:", len(error.metadata["attempts"]))
        print("calls:", calls)

        print()
        print("attempts:")

        for index, attempt in enumerate(
            error.metadata["attempts"],
            start=1,
        ):
            print(
                index,
                attempt,
            )

finally:
    provider.chat = original_chat
