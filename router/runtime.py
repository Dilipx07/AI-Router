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
from .models import MODEL_REGISTRY
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

    _started: bool = False

    _shutdown: bool = False

    # ========================================================
    # STARTUP
    # ========================================================

    def startup(self) -> None:
        """
        Start the runtime.

        Startup is idempotent.
        """

        if self._shutdown:

            raise RuntimeError(
                "RouterRuntime has already been shut down."
            )

        if self._started:

            return

        if self.persistence is not None:

            self.persistence.startup()

        self._started = True

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:
        """
        Shut down the runtime cleanly.

        Shutdown is idempotent.
        """

        if self._shutdown:

            return

        if self.persistence is not None:

            self.persistence.shutdown()

        self._shutdown = True

        self._started = False

    # ========================================================
    # STATE
    # ========================================================

    @property
    def started(self) -> bool:
        return self._started

    @property
    def shutdown_complete(self) -> bool:
        return self._shutdown


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
        model_registry=MODEL_REGISTRY,
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

    runtime = RouterRuntime(
        config=config,
        policy=policy,
        persistence=persistence,
        metrics=metrics,
        quota=quota,
        health=health,
        engine=engine,
        executor=executor,
    )

    runtime.startup()

    return runtime


# ============================================================
# DEFAULT RUNTIME
# ============================================================

RUNTIME = create_runtime()
