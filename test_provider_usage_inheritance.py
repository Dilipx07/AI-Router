from providers.groq import GroqProvider
from providers.openrouter import OpenRouterProvider
from providers.huggingface import HuggingFaceProvider
from providers.openai_compatible import OpenAICompatibleProvider


EXPECTED = {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15,
}


response = {
    "usage": EXPECTED,
}


providers = [
    (
        "groq",
        GroqProvider(
            config={
                "api_key": "test",
                "base_url": "https://example.com/v1",
            }
        ),
    ),
    (
        "openrouter",
        OpenRouterProvider(
            config={
                "api_key": "test",
                "base_url": "https://example.com/v1",
            }
        ),
    ),
    (
        "huggingface",
        HuggingFaceProvider(
            config={
                "api_key": "test",
                "base_url": "https://example.com/v1",
            }
        ),
    ),
]


for name, provider in providers:

    usage = provider._extract_usage(
        response
    )

    print(
        name,
        "→",
        usage,
    )

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


# ------------------------------------------------------------
# Verify all three actually inherit the shared implementation.
# ------------------------------------------------------------

assert (
    GroqProvider._extract_usage
    is OpenAICompatibleProvider._extract_usage
)

assert (
    OpenRouterProvider._extract_usage
    is OpenAICompatibleProvider._extract_usage
)

assert (
    HuggingFaceProvider._extract_usage
    is OpenAICompatibleProvider._extract_usage
)


print()
print("shared usage extractor: PASS")
print("Groq: PASS")
print("OpenRouter: PASS")
print("Hugging Face: PASS")
print()
print("RESULT: PASS")
