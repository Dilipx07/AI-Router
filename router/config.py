"""
AI Router v1
Runtime configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .policy import POLICIES


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RouterConfig:
    """
    Immutable runtime configuration.
    """

    policy: str = "balanced"

    request_timeout: float = 60.0

    recent_failure_window: float = 300.0

    persistence_enabled: bool = True


# ============================================================
# ENVIRONMENT PARSERS
# ============================================================

def _env_bool(
    name: str,
    default: bool,
) -> bool:
    """
    Read a boolean environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    return normalized in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_float(
    name: str,
    default: float,
) -> float:
    """
    Read and validate a non-negative float environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = float(value)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f"Invalid numeric value for {name}: "
            f"{value!r}"
        ) from exc

    if parsed < 0:

        raise ValueError(
            f"{name} must be >= 0"
        )

    return parsed


# ============================================================
# CONFIGURATION LOADER
# ============================================================

def load_config() -> RouterConfig:
    """
    Load and validate runtime configuration.
    """

    policy = (
        os.getenv(
            "AI_ROUTER_POLICY",
            "balanced",
        )
        .strip()
        .lower()
    )

    if not policy:

        policy = "balanced"

    if policy not in POLICIES:

        available = ", ".join(
            sorted(POLICIES.keys())
        )

        raise ValueError(
            f"Unknown AI_ROUTER_POLICY: "
            f"{policy!r}. "
            f"Available policies: {available}"
        )

    request_timeout = _env_float(
        "AI_ROUTER_REQUEST_TIMEOUT",
        60.0,
    )

    recent_failure_window = _env_float(
        "AI_ROUTER_RECENT_FAILURE_WINDOW",
        300.0,
    )

    persistence_enabled = _env_bool(
        "AI_ROUTER_PERSISTENCE_ENABLED",
        True,
    )

    return RouterConfig(
        policy=policy,

        request_timeout=request_timeout,

        recent_failure_window=(
            recent_failure_window
        ),

        persistence_enabled=(
            persistence_enabled
        ),
    )


# ============================================================
# DEFAULT CONFIG
# ============================================================

CONFIG = load_config()
