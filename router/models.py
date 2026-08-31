"""
Model intelligence layer for AI Router v1.

This module converts provider configuration into normalized
model candidates that can later be scored and routed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from config.registry import (
    ModelConfig,
    ProviderConfig,
    PROVIDER_REGISTRY,
)


# ============================================================
# MODEL CANDIDATE
# ============================================================

@dataclass(frozen=True)
class ModelCandidate:
    """
    Normalized model candidate used by the routing engine.

    A candidate represents one concrete provider + model pair.
    """

    provider: str

    model: str

    priority: int = 100

    enabled: bool = True

    capabilities: frozenset[str] = field(
        default_factory=frozenset
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def identifier(self) -> str:
        """Unique human-readable candidate identifier."""
        return f"{self.provider}/{self.model}"

    def supports(
        self,
        capability: str,
    ) -> bool:
        """Return whether this model supports a capability."""
        return capability in self.capabilities


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:
    """
    Runtime model registry.

    Builds a normalized index of all provider/model combinations.
    """

    def __init__(
        self,
        providers: Optional[
            Iterable[ProviderConfig]
        ] = None,
    ) -> None:

        if providers is None:
            providers = PROVIDER_REGISTRY.values()

        self._candidates: list[ModelCandidate] = []

        for provider in providers:
            self._register_provider(provider)

    # --------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------

    def _register_provider(
        self,
        provider: ProviderConfig,
    ) -> None:

        for model in provider.models:

            candidate = ModelCandidate(
                provider=provider.name,
                model=model.name,
                priority=model.priority,
                enabled=(
                    provider.enabled
                    and model.enabled
                ),
                capabilities=model.capabilities,
                metadata={
                    **provider.metadata,
                    **model.metadata,
                },
            )

            self._candidates.append(candidate)

    # --------------------------------------------------------
    # ALL MODELS
    # --------------------------------------------------------

    def all(
        self,
        *,
        enabled_only: bool = True,
    ) -> list[ModelCandidate]:
        """
        Return all registered model candidates.
        """

        candidates = self._candidates

        if enabled_only:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.enabled
            ]

        return list(candidates)

    # --------------------------------------------------------
    # PROVIDER FILTER
    # --------------------------------------------------------

    def by_provider(
        self,
        provider: str,
        *,
        enabled_only: bool = True,
    ) -> list[ModelCandidate]:
        """
        Return candidates belonging to one provider.
        """

        return [
            candidate
            for candidate in self.all(
                enabled_only=enabled_only
            )
            if candidate.provider == provider
        ]

    # --------------------------------------------------------
    # CAPABILITY FILTER
    # --------------------------------------------------------

    def by_capability(
        self,
        capability: str,
        *,
        enabled_only: bool = True,
    ) -> list[ModelCandidate]:
        """
        Return candidates supporting a capability.
        """

        return [
            candidate
            for candidate in self.all(
                enabled_only=enabled_only
            )
            if candidate.supports(capability)
        ]

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    def get(
        self,
        provider: str,
        model: str,
    ) -> Optional[ModelCandidate]:
        """
        Find one exact provider/model pair.
        """

        for candidate in self._candidates:

            if (
                candidate.provider == provider
                and candidate.model == model
            ):
                return candidate

        return None

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query: str,
    ) -> list[ModelCandidate]:
        """
        Search providers and model names.

        Matching is case-insensitive.
        """

        query = query.strip().lower()

        if not query:
            return []

        return [
            candidate
            for candidate in self.all()
            if (
                query in candidate.provider.lower()
                or query in candidate.model.lower()
                or query in candidate.identifier.lower()
            )
        ]

    # --------------------------------------------------------
    # PROVIDER LIST
    # --------------------------------------------------------

    def providers(self) -> list[str]:
        """
        Return unique provider names represented in the registry.
        """

        result = []
        seen = set()

        for candidate in self._candidates:

            if candidate.provider in seen:
                continue

            seen.add(candidate.provider)
            result.append(candidate.provider)

        return result

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    def count(
        self,
        *,
        enabled_only: bool = True,
    ) -> int:
        """Return the number of registered candidates."""

        return len(
            self.all(
                enabled_only=enabled_only
            )
        )


# ============================================================
# DEFAULT REGISTRY
# ============================================================

MODEL_REGISTRY = ModelRegistry()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_model(
    provider: str,
    model: str,
) -> Optional[ModelCandidate]:
    """Get one model candidate."""

    return MODEL_REGISTRY.get(
        provider,
        model,
    )


def get_all_models() -> list[ModelCandidate]:
    """Get every enabled model candidate."""

    return MODEL_REGISTRY.all()


def get_provider_models(
    provider: str,
) -> list[ModelCandidate]:
    """Get all enabled models for a provider."""

    return MODEL_REGISTRY.by_provider(
        provider
    )


def get_capable_models(
    capability: str,
) -> list[ModelCandidate]:
    """Get models supporting a capability."""

    return MODEL_REGISTRY.by_capability(
        capability
    )