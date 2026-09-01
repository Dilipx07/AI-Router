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

    metrics = MetricsStore(
        persistence=persistence
    )

    quota = QuotaStore(
        persistence=persistence
    )

    metrics.record_success(
        "ollama",
        "auto-persist-model",
        usage=ProviderUsage(
            prompt_tokens=21,
            completion_tokens=4,
            total_tokens=25,
        ),
        elapsed=2.5,
    )

    quota.record(
        "ollama",
        "auto-persist-model",
        usage=ProviderUsage(
            prompt_tokens=30,
            completion_tokens=5,
            total_tokens=35,
        ),
    )

    # --------------------------------------------------------
    # Create a new persistence reader to simulate restart.
    # --------------------------------------------------------

    restored = RuntimePersistence(
        db_path=db_path
    )

    metrics_rows = restored.load_metrics()
    quota_rows = restored.load_quota()

    print("=== AUTOMATIC PERSISTENCE TEST ===")

    print(
        "metrics rows:",
        metrics_rows,
    )

    print(
        "quota rows:",
        quota_rows,
    )

    assert len(metrics_rows) == 1
    assert len(quota_rows) == 1

    metric = metrics_rows[0]
    state = quota_rows[0]

    assert metric.provider == "ollama"
    assert metric.model == "auto-persist-model"
    assert metric.total_tokens == 25
    assert metric.requests == 1
    assert metric.successes == 1

    assert state.provider == "ollama"
    assert state.model == "auto-persist-model"
    assert state.total_tokens == 35
    assert state.requests == 1

    print()
    print("metrics auto-persist: PASS")
    print("quota auto-persist: PASS")
    print("RESULT: PASS")
