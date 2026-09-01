from router.runtime import create_runtime


runtime = create_runtime()


print("=== RUNTIME BOOTSTRAP TEST ===")


assert runtime.persistence is not None
assert runtime.metrics is not None
assert runtime.quota is not None
assert runtime.health is not None
assert runtime.engine is not None
assert runtime.executor is not None


# ------------------------------------------------------------
# Shared metrics instance
# ------------------------------------------------------------

assert (
    runtime.executor.metrics_store
    is runtime.metrics
)

assert (
    runtime.engine.metrics_store
    is runtime.metrics
)


# ------------------------------------------------------------
# Shared quota instance
# ------------------------------------------------------------

assert (
    runtime.executor.quota_store
    is runtime.quota
)

assert (
    runtime.engine.quota_store
    is runtime.quota
)


# ------------------------------------------------------------
# Shared health instance
# ------------------------------------------------------------

assert (
    runtime.executor.health_store
    is runtime.health
)

assert (
    runtime.engine.health_store
    is runtime.health
)


# ------------------------------------------------------------
# Persistence wiring
# ------------------------------------------------------------

assert (
    runtime.metrics.persistence
    is runtime.persistence
)

assert (
    runtime.quota.persistence
    is runtime.persistence
)


# ------------------------------------------------------------
# Provider wiring
# ------------------------------------------------------------

assert runtime.executor.providers

print("persistence shared: PASS")
print("metrics shared: PASS")
print("quota shared: PASS")
print("health shared: PASS")
print("engine/executor wiring: PASS")
print("provider wiring: PASS")

print()
print("RESULT: PASS")
