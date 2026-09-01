from router.runtime import create_runtime


print("=== RUNTIME LIFECYCLE TEST ===")


runtime = create_runtime()


print(
    "initial started:",
    runtime.started,
)

assert runtime.started is True
assert runtime.shutdown_complete is False


# ------------------------------------------------------------
# Startup idempotency
# ------------------------------------------------------------

runtime.startup()

assert runtime.started is True

print(
    "startup idempotency: PASS"
)


# ------------------------------------------------------------
# Persistence state
# ------------------------------------------------------------

if runtime.persistence is not None:

    assert (
        runtime.persistence.closed
        is False
    )

print(
    "persistence active: PASS"
)


# ------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------

runtime.shutdown()

assert runtime.started is False

assert (
    runtime.shutdown_complete
    is True
)

print(
    "shutdown: PASS"
)


# ------------------------------------------------------------
# Shutdown idempotency
# ------------------------------------------------------------

runtime.shutdown()

assert (
    runtime.shutdown_complete
    is True
)

print(
    "shutdown idempotency: PASS"
)


# ------------------------------------------------------------
# Restart must fail after shutdown
# ------------------------------------------------------------

try:

    runtime.startup()

except RuntimeError:

    pass

else:

    raise AssertionError(
        "startup after shutdown must fail"
    )


print(
    "restart-after-shutdown protection: PASS"
)


# ------------------------------------------------------------
# Persistence must reject writes after close
# ------------------------------------------------------------

if runtime.persistence is not None:

    try:

        runtime.persistence.load_metrics()

    except RuntimeError:

        pass

    else:

        raise AssertionError(
            "closed persistence must reject operations"
        )

    print(
        "closed persistence protection: PASS"
    )


print()
print("RESULT: PASS")
