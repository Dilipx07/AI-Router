from providers.base import ProviderUsage
from router.metrics import MetricsStore


metrics = MetricsStore()


metrics.record_success(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
    ),
    elapsed=1.5,
)

metrics.record_success(
    "ollama",
    "nemotron-3-nano:30b-cloud",
    usage=ProviderUsage(
        prompt_tokens=20,
        completion_tokens=3,
        total_tokens=23,
    ),
    elapsed=2.0,
)

metrics.record_success(
    "google",
    "gemini-3.1-flash-lite",
    usage=ProviderUsage(
        prompt_tokens=30,
        completion_tokens=5,
        total_tokens=35,
    ),
    elapsed=1.0,
)


# ------------------------------------------------------------
# All metrics
# ------------------------------------------------------------

all_records = metrics.snapshot()

print("=== METRICS SNAPSHOT TEST ===")
print("all records:", len(all_records))

assert len(all_records) == 3


# ------------------------------------------------------------
# Provider snapshot
# ------------------------------------------------------------

ollama_records = metrics.by_provider(
    "ollama"
)

print("ollama records:", len(ollama_records))

assert len(ollama_records) == 2

assert all(
    record.provider == "ollama"
    for record in ollama_records
)


# ------------------------------------------------------------
# Exact model lookup
# ------------------------------------------------------------

gemma = metrics.by_model(
    "ollama",
    "gemma4:31b-cloud",
)

assert gemma is not None

assert gemma.total_tokens == 12


missing = metrics.by_model(
    "ollama",
    "does-not-exist",
)

assert missing is None


# ------------------------------------------------------------
# Reset one provider
# ------------------------------------------------------------

metrics.reset(
    provider="ollama",
)

assert metrics.by_model(
    "ollama",
    "gemma4:31b-cloud",
) is None

assert metrics.by_model(
    "ollama",
    "nemotron-3-nano:30b-cloud",
) is None

assert metrics.by_model(
    "google",
    "gemini-3.1-flash-lite",
) is not None


print("provider filtering: PASS")
print("exact model lookup: PASS")
print("provider reset isolation: PASS")
print()
print("RESULT: PASS")
