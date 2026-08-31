from concurrent.futures import ThreadPoolExecutor

from providers.base import ProviderUsage

from router.metrics import MetricsStore
from router.quota import QuotaStore


WORKERS = 20
REQUESTS_PER_WORKER = 50

TOTAL = (
    WORKERS
    * REQUESTS_PER_WORKER
)


metrics = MetricsStore()
quota = QuotaStore()


def worker(index: int):

    for _ in range(
        REQUESTS_PER_WORKER
    ):

        metrics.record_success(
            "ollama",
            "concurrent-model",
            usage=ProviderUsage(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
            elapsed=0.01,
        )

        quota.record(
            "ollama",
            "concurrent-model",
            usage=ProviderUsage(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            ),
        )

    return index


print("=== CONCURRENCY TEST ===")


with ThreadPoolExecutor(
    max_workers=WORKERS
) as executor:

    results = list(
        executor.map(
            worker,
            range(WORKERS),
        )
    )


assert len(results) == WORKERS


metric = metrics.get(
    "ollama",
    "concurrent-model",
)

state = quota.get(
    "ollama",
    "concurrent-model",
)


print(
    "expected requests:",
    TOTAL,
)

print(
    "metrics requests:",
    metric.requests,
)

print(
    "metrics successes:",
    metric.successes,
)

print(
    "metrics failures:",
    metric.failures,
)

print(
    "metrics total tokens:",
    metric.total_tokens,
)

print(
    "quota requests:",
    state.requests,
)

print(
    "quota total tokens:",
    state.total_tokens,
)


assert metric.requests == TOTAL
assert metric.successes == TOTAL
assert metric.failures == 0

assert metric.prompt_tokens == TOTAL
assert metric.completion_tokens == TOTAL
assert metric.total_tokens == (
    TOTAL * 2
)

assert state.requests == TOTAL
assert state.prompt_tokens == TOTAL
assert state.completion_tokens == TOTAL
assert state.total_tokens == (
    TOTAL * 2
)


print()
print("metrics thread safety: PASS")
print("quota thread safety: PASS")
print("counter integrity: PASS")
print()
print("RESULT: PASS")
