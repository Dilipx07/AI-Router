"""
AI Router v1
Routing policy definitions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingPolicy:
    """
    Runtime scoring weights for one routing strategy.
    """

    name: str

    latency_multiplier: float = 1.0

    reliability_multiplier: float = 1.0

    recent_failure_multiplier: float = 1.0

    cost_multiplier: float = 1.0

    quota_multiplier: float = 1.0


POLICIES: dict[str, RoutingPolicy] = {

    "balanced": RoutingPolicy(
        name="balanced",
        latency_multiplier=1.0,
        reliability_multiplier=1.0,
        recent_failure_multiplier=1.0,
        cost_multiplier=1.0,
        quota_multiplier=1.0,
    ),

    "fast": RoutingPolicy(
        name="fast",
        latency_multiplier=2.0,
        reliability_multiplier=0.75,
        recent_failure_multiplier=0.75,
        cost_multiplier=0.5,
        quota_multiplier=0.75,
    ),

    "reliable": RoutingPolicy(
        name="reliable",
        latency_multiplier=0.75,
        reliability_multiplier=2.0,
        recent_failure_multiplier=2.0,
        cost_multiplier=0.5,
        quota_multiplier=0.75,
    ),

    "cheap": RoutingPolicy(
        name="cheap",
        latency_multiplier=0.5,
        reliability_multiplier=0.75,
        recent_failure_multiplier=0.75,
        cost_multiplier=2.0,
        quota_multiplier=1.25,
    ),
}


def get_policy(
    name: str | None,
) -> RoutingPolicy:

    if not name:
        return POLICIES["balanced"]

    normalized = name.strip().lower()

    return POLICIES.get(
        normalized,
        POLICIES["balanced"],
    )
