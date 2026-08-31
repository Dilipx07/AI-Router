from router.executor import RouterExecutor, ExecutionError
from router.engine import RoutingDecision
from router.health import HealthStore
from providers.base import ProviderRequest, ProviderResult
from providers.factory import PROVIDERS
from router.errors import RouterError, ErrorType


provider = PROVIDERS["ollama"]
original_chat = provider.chat

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: OK",
        }
    ]
)


def success(model):
    return ProviderResult.success_result(
        provider="ollama",
        model=model,
        content="OK",
    )


def run_case(name, error_type, retryable, explicit_model=False):
    calls = []

    def fake_chat(model, request):
        calls.append(model)

        if model == "gemma4:31b-cloud":
            raise RouterError(
                error_type,
                f"Simulated {error_type.value}",
                provider="ollama",
                model=model,
                retryable=retryable,
            )

        return success(model)

    provider.chat = fake_chat

    try:
        executor = RouterExecutor(
            providers=PROVIDERS,
            health_store=HealthStore(),
        )

        result = executor.route_and_execute(
            request,
            provider="ollama",
            model=(
                "gemma4:31b-cloud"
                if explicit_model
                else None
            ),
        )

        print(
            f"{name}: PASS | "
            f"result={result.model} | "
            f"calls={calls}"
        )

    except ExecutionError as error:
        print(
            f"{name}: STOP | "
            f"type={error.error_type.value} | "
            f"calls={calls}"
        )

    finally:
        provider.chat = original_chat


print("=" * 70)
print("EXECUTOR FAILURE MATRIX")
print("=" * 70)

run_case(
    "RATE_LIMIT",
    ErrorType.RATE_LIMIT,
    True,
)

run_case(
    "QUOTA_EXCEEDED",
    ErrorType.QUOTA_EXCEEDED,
    True,
)

run_case(
    "PAYMENT_REQUIRED",
    ErrorType.PAYMENT_REQUIRED,
    False,
)

run_case(
    "AUTH_FAILURE",
    ErrorType.AUTH_FAILURE,
    False,
)

run_case(
    "MODEL_NOT_FOUND automatic",
    ErrorType.MODEL_NOT_FOUND,
    False,
)

run_case(
    "MODEL_NOT_FOUND explicit",
    ErrorType.MODEL_NOT_FOUND,
    False,
    explicit_model=True,
)

run_case(
    "CONTEXT_TOO_LARGE",
    ErrorType.CONTEXT_TOO_LARGE,
    False,
)

run_case(
    "TIMEOUT",
    ErrorType.TIMEOUT,
    True,
)

run_case(
    "NETWORK_FAILURE",
    ErrorType.NETWORK_FAILURE,
    True,
)

run_case(
    "EMPTY_RESPONSE",
    ErrorType.EMPTY_RESPONSE,
    True,
)

print("=" * 70)
