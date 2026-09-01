import os

from router.config import load_config


print("=== CONFIGURATION TEST ===")


ENV_NAMES = (
    "AI_ROUTER_POLICY",
    "AI_ROUTER_REQUEST_TIMEOUT",
    "AI_ROUTER_RECENT_FAILURE_WINDOW",
    "AI_ROUTER_PERSISTENCE_ENABLED",
)


# ============================================================
# CLEAN ENVIRONMENT
# ============================================================

for name in ENV_NAMES:
    os.environ.pop(
        name,
        None,
    )


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

config = load_config()

print(
    "default policy:",
    config.policy,
)

print(
    "default timeout:",
    config.request_timeout,
)

print(
    "default failure window:",
    config.recent_failure_window,
)

print(
    "default persistence:",
    config.persistence_enabled,
)


assert config.policy == "balanced"

assert config.request_timeout == 60.0

assert (
    config.recent_failure_window
    == 300.0
)

assert (
    config.persistence_enabled
    is True
)

print("defaults: PASS")


# ============================================================
# CUSTOM ENVIRONMENT
# ============================================================

os.environ[
    "AI_ROUTER_POLICY"
] = "fast"

os.environ[
    "AI_ROUTER_REQUEST_TIMEOUT"
] = "45"

os.environ[
    "AI_ROUTER_RECENT_FAILURE_WINDOW"
] = "120"

os.environ[
    "AI_ROUTER_PERSISTENCE_ENABLED"
] = "false"


config = load_config()


assert config.policy == "fast"

assert config.request_timeout == 45.0

assert (
    config.recent_failure_window
    == 120.0
)

assert (
    config.persistence_enabled
    is False
)

print(
    "environment configuration: PASS"
)


# ============================================================
# INVALID TIMEOUT
# ============================================================

os.environ[
    "AI_ROUTER_REQUEST_TIMEOUT"
] = "invalid"


try:

    load_config()

except ValueError:

    pass

else:

    raise AssertionError(
        "Invalid timeout must raise ValueError"
    )


print(
    "invalid numeric configuration: PASS"
)


# ============================================================
# NEGATIVE TIMEOUT
# ============================================================

os.environ[
    "AI_ROUTER_REQUEST_TIMEOUT"
] = "-1"


try:

    load_config()

except ValueError:

    pass

else:

    raise AssertionError(
        "Negative timeout must raise ValueError"
    )


print(
    "negative numeric configuration: PASS"
)


# ============================================================
# INVALID POLICY
# ============================================================

os.environ[
    "AI_ROUTER_REQUEST_TIMEOUT"
] = "60"

os.environ[
    "AI_ROUTER_POLICY"
] = "does-not-exist"


try:

    load_config()

except ValueError:

    pass

else:

    raise AssertionError(
        "Invalid policy must raise ValueError"
    )


print(
    "invalid policy: PASS"
)


# ============================================================
# CLEANUP
# ============================================================

for name in ENV_NAMES:
    os.environ.pop(
        name,
        None,
    )


print()
print("RESULT: PASS")
