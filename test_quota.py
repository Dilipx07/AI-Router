from providers.base import ProviderUsage
from router.quota import (
    QuotaLimit,
    QuotaStore,
)


quota = QuotaStore()


quota.set_limit(
    "ollama",
    "gemma4:31b-cloud",
    QuotaLimit(
        max_requests=3,
        max_tokens=100,
    ),
)


# ------------------------------------------------------------
# Initial state
# ------------------------------------------------------------

status = quota.check(
    "ollama",
    "gemma4:31b-cloud",
)

print("=== QUOTA TEST ===")
print("initial allowed:", status.allowed)

assert status.allowed
assert status.remaining_requests == 3
assert status.remaining_tokens == 100


# ------------------------------------------------------------
# Request 1
# ------------------------------------------------------------

quota.record(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    ),
)

status = quota.check(
    "ollama",
    "gemma4:31b-cloud",
)

print("after request 1:", status)

assert status.allowed
assert status.remaining_requests == 2
assert status.remaining_tokens == 70


# ------------------------------------------------------------
# Request 2
# ------------------------------------------------------------

quota.record(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=20,
        completion_tokens=15,
        total_tokens=35,
    ),
)

status = quota.check(
    "ollama",
    "gemma4:31b-cloud",
)

print("after request 2:", status)

assert status.allowed
assert status.remaining_requests == 1
assert status.remaining_tokens == 35


# ------------------------------------------------------------
# Request 3
# ------------------------------------------------------------

quota.record(
    "ollama",
    "gemma4:31b-cloud",
    usage=ProviderUsage(
        prompt_tokens=20,
        completion_tokens=15,
        total_tokens=35,
    ),
)

status = quota.check(
    "ollama",
    "gemma4:31b-cloud",
)

print("after request 3:", status)

assert not status.allowed
assert status.reason == "request_quota_exceeded"
assert status.remaining_requests == 0


state = quota.get(
    "ollama",
    "gemma4:31b-cloud",
)

assert state.requests == 3
assert state.prompt_tokens == 60
assert state.completion_tokens == 40
assert state.total_tokens == 100


# ------------------------------------------------------------
# Unlimited model
# ------------------------------------------------------------

unlimited = quota.check(
    "ollama",
    "no-limit-model",
)

assert unlimited.allowed


print()
print("request quota: PASS")
print("token quota: PASS")
print("usage accumulation: PASS")
print("unlimited model: PASS")
print()
print("RESULT: PASS")
