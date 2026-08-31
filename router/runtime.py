"""
AI Router v1
Runtime/bootstrap container.

Creates and wires the complete production router runtime
using explicit dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass

from providers.factory import PROVIDERS

from .config import (
    RouterConfig,
    load_config,
)
from .engine import RouterEngine
from .executor import RouterExecutor
from .health import HealthStore
from .metrics import MetricsStore
from .persistence import RuntimePersistence
from .policy import (
    RoutingPolicy,
    get_policy,
)
from .quota import QuotaStore


# ============================================================
# RUNTIME
# ============================================================

@dataclass
class RouterRuntime:
    """
    Fully wired router runtime.

    Every stateful component belongs to this runtime.
    """

    config: RouterConfig

    policy: RoutingPolicy

    persistence: RuntimePersistence | None

    metrics: MetricsStore

    quota: QuotaStore

    health: HealthStore

    engine: RouterEngine

    executor: RouterExecutor


# ============================================================
# FACTORY
# ============================================================

def create_runtime() -> RouterRuntime:
    """
    Create a fully isolated router runtime.
    """

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config()

    policy = get_policy(
        config.policy
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistence = None

    if config.persistence_enabled:

        persistence = RuntimePersistence()

    # --------------------------------------------------------
    # Runtime stores
    # --------------------------------------------------------

    metrics = MetricsStore(
        persistence=persistence,
    )

    quota = QuotaStore(
        persistence=persistence,
    )

    health = HealthStore()

    # --------------------------------------------------------
    # Router engine
    # --------------------------------------------------------

    engine = RouterEngine(
        health_store=health,
        quota_store=quota,
        metrics_store=metrics,
        policy=policy,
    )

    # --------------------------------------------------------
    # Executor
    # --------------------------------------------------------

    executor = RouterExecutor(
        providers=PROVIDERS,
        health_store=health,
        metrics_store=metrics,
        quota_store=quota,
        router=engine,
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    return RouterRuntime(
        config=config,

        policy=policy,

        persistence=persistence,

        metrics=metrics,

        quota=quota,

        health=health,

        engine=engine,

        executor=executor,
    )


# ============================================================
# DEFAULT RUNTIME
# ============================================================

RUNTIME = create_runtime()
