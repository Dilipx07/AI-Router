from providers.base import ProviderUsage
from router.metrics import MetricsStore


metrics = MetricsStore()


success = metrics.record_success(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=21,
        completion_tokens=4,
        total_tokens=25,
    ),
    elapsed=2.5,
)

failure = metrics.record_failure(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
    ),
    elapsed=1.5,
    error_type="timeout",
)


record = metrics.get(
    "ollama",
    "gemma4:31b-cloud",
)

print("=== METRICS MODEL TEST ===")
print("provider:", record.provider)
print("model:", record.model)
print("requests:", record.requests)
print("successes:", record.successes)
print("failures:", record.failures)
print("prompt_tokens:", record.prompt_tokens)
print("completion_tokens:", record.completion_tokens)
print("total_tokens:", record.total_tokens)
print("total_latency:", record.total_latency)
print("average_latency:", record.average_latency)
print("last_latency:", record.last_latency)
print("success_rate:", record.success_rate)
print("failure_rate:", record.failure_rate)
print("last_error_type:", record.last_error_type)

assert record.requests == 2
assert record.successes == 1
assert record.failures == 1

assert record.prompt_tokens == 31
assert record.completion_tokens == 6
assert record.total_tokens == 37

assert record.total_latency == 4.0
assert record.last_latency == 1.5

assert record.success_rate == 0.5
assert record.failure_rate == 0.5

assert record.last_error_type == "timeout"

print()
print("RESULT: PASS")
