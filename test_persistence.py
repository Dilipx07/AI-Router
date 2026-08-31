import tempfile
from pathlib import Path

from providers.base import ProviderUsage

from router.metrics import MetricsStore
from router.quota import QuotaStore
from router.persistence import RuntimePersistence


with tempfile.TemporaryDirectory() as temp_dir:

    db_path = Path(temp_dir) / "runtime.db"

    persistence = RuntimePersistence(
        db_path=db_path
    )

    metrics = MetricsStore()

    metrics.record_success(
        "ollama",
        "persist-model",
        usage=ProviderUsage(
            prompt_tokens=21,
            completion_tokens=4,
            total_tokens=25,
        ),
        elapsed=2.5,
    )

    record = metrics.get(
        "ollama",
        "persist-model",
    )

    persistence.save_metrics(record)

    quota = QuotaStore()

    quota.record(
        "ollama",
        "persist-model",
        usage=ProviderUsage(
            prompt_tokens=30,
            completion_tokens=5,
            total_tokens=35,
        ),
    )

    state = quota.get(
        "ollama",
        "persist-model",
    )

    persistence.save_quota(state)


    # --------------------------------------------------------
    # Simulate restart by creating a new persistence instance.
    # --------------------------------------------------------

    restored = RuntimePersistence(
        db_path=db_path
    )

    restored_metrics = restored.load_metrics()

    assert len(restored_metrics) == 1

    restored_metric = restored_metrics[0]

    assert restored_metric.provider == "ollama"
    assert restored_metric.model == "persist-model"
    assert restored_metric.requests == 1
    assert restored_metric.successes == 1
    assert restored_metric.failures == 0
    assert restored_metric.prompt_tokens == 21
    assert restored_metric.completion_tokens == 4
    assert restored_metric.total_tokens == 25
    assert restored_metric.total_latency == 2.5


    restored_quota = restored.load_quota()

    assert len(restored_quota) == 1

    restored_state = restored_quota[0]

    assert restored_state.provider == "ollama"
    assert restored_state.model == "persist-model"
    assert restored_state.requests == 1
    assert restored_state.prompt_tokens == 30
    assert restored_state.completion_tokens == 5
    assert restored_state.total_tokens == 35


    print("=== PERSISTENCE TEST ===")
    print("database:", db_path)
    print("metrics restored:", restored_metrics)
    print("quota restored:", restored_quota)
    print()
    print("metrics persistence: PASS")
    print("quota persistence: PASS")
    print("restart restoration: PASS")
    print()
    print("RESULT: PASS")
