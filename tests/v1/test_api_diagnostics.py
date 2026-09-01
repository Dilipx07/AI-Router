from fastapi.testclient import TestClient

import router_api

from providers.base import ProviderUsage
from router.metrics import METRICS
from router.quota import QUOTA


# ------------------------------------------------------------
# Seed runtime state
# ------------------------------------------------------------

METRICS.reset()
QUOTA.reset()


METRICS.record_success(
    "ollama",
    "metrics-api-model",
    usage=ProviderUsage(
        prompt_tokens=21,
        completion_tokens=4,
        total_tokens=25,
    ),
    elapsed=2.5,
)


QUOTA.record(
    "ollama",
    "quota-api-model",
    usage=ProviderUsage(
        prompt_tokens=30,
        completion_tokens=5,
        total_tokens=35,
    ),
)


client = TestClient(
    router_api.app
)


# ------------------------------------------------------------
# Metrics endpoint
# ------------------------------------------------------------

metrics_response = client.get(
    "/metrics"
)

print("=== API METRICS TEST ===")
print("status:", metrics_response.status_code)

assert metrics_response.status_code == 200

metrics_data = metrics_response.json()

print("response:", metrics_data)

assert metrics_data["object"] == "metrics"

assert len(
    metrics_data["data"]
) == 1

metric = metrics_data["data"][0]

assert metric["provider"] == "ollama"
assert metric["model"] == "metrics-api-model"

assert metric["requests"] == 1
assert metric["successes"] == 1
assert metric["failures"] == 0

assert metric["prompt_tokens"] == 21
assert metric["completion_tokens"] == 4
assert metric["total_tokens"] == 25


# ------------------------------------------------------------
# Quota endpoint
# ------------------------------------------------------------

quota_response = client.get(
    "/quota"
)

print()
print("=== API QUOTA TEST ===")
print("status:", quota_response.status_code)

assert quota_response.status_code == 200

quota_data = quota_response.json()

print("response:", quota_data)

assert quota_data["object"] == "quota"

assert len(
    quota_data["data"]
) == 1

state = quota_data["data"][0]

assert state["provider"] == "ollama"
assert state["model"] == "quota-api-model"

assert state["requests"] == 1
assert state["prompt_tokens"] == 30
assert state["completion_tokens"] == 5
assert state["total_tokens"] == 35


# ------------------------------------------------------------
# Verify read-only behavior
# ------------------------------------------------------------

assert METRICS.get(
    "ollama",
    "metrics-api-model",
).requests == 1

assert QUOTA.get(
    "ollama",
    "quota-api-model",
).requests == 1


print()
print("metrics endpoint: PASS")
print("quota endpoint: PASS")
print("read-only behavior: PASS")
print()
print("RESULT: PASS")
