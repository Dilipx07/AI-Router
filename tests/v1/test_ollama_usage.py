from providers.ollama import OllamaProvider
from providers.base import ProviderRequest


provider = OllamaProvider(
    config={
        "base_url": "http://localhost:11434",
        "timeout": 120,
    }
)

request = ProviderRequest(
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: USAGE_OK",
        }
    ],
    max_tokens=20,
)

result = provider.chat(
    model="gemma4:31b-cloud",
    request=request,
)

print("=== OLLAMA USAGE TEST ===")
print("success:", result.success)
print("provider:", result.provider)
print("model:", result.model)
print("content:", result.content)
print("status:", result.status)
print("usage:", result.usage)
print("prompt_tokens:", result.usage.prompt_tokens)
print("completion_tokens:", result.usage.completion_tokens)
print("total_tokens:", result.usage.total_tokens)
print("usage_known:", result.usage.known)

assert result.success
assert result.content.strip() == "USAGE_OK"

assert result.usage.prompt_tokens >= 0
assert result.usage.completion_tokens >= 0
assert result.usage.total_tokens >= 0

assert (
    result.usage.total_tokens
    >= result.usage.prompt_tokens
)

assert (
    result.usage.total_tokens
    >= result.usage.completion_tokens
)

print()
print("RESULT: PASS")
