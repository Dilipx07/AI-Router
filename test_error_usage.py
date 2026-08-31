from providers.base import ProviderUsage
from router.errors import RouterError, ErrorType


error = RouterError(
    ErrorType.PROVIDER_FAILURE,
    "Simulated failure",
    provider="ollama",
    model="test-model",
    retryable=True,
    usage=ProviderUsage(
        prompt_tokens=21,
        completion_tokens=4,
        total_tokens=25,
    ),
)

print("=== ERROR USAGE TEST ===")
print("error:", error)
print("usage:", error.usage)
print("prompt_tokens:", error.usage.prompt_tokens)
print("completion_tokens:", error.usage.completion_tokens)
print("total_tokens:", error.usage.total_tokens)

assert error.usage.prompt_tokens == 21
assert error.usage.completion_tokens == 4
assert error.usage.total_tokens == 25

print()
print("usage preserved on RouterError: True")
print("RESULT: PASS")
