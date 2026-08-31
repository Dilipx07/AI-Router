"""
AI Router v1
Provider factory.

Creates provider adapters from the central provider registry.
The router should use this factory instead of importing concrete
provider implementations directly.
"""

from __future__ import annotations

from typing import Any, Optional

from config.registry import PROVIDER_REGISTRY
from .base import BaseProvider
from .cloudflare import CloudflareProvider
from .groq import GroqProvider
from .huggingface import HuggingFaceProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider
from .google import GoogleProvider


# ============================================================
# PROVIDER CLASS REGISTRY
# ============================================================

PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "google": GoogleProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "huggingface": HuggingFaceProvider,
    "ollama": OllamaProvider,
}


# ============================================================
# FACTORY
# ============================================================

def create_provider(
    provider_name: str,
    *,
    config: Optional[dict[str, Any]] = None,
) -> BaseProvider:
    """
    Create a concrete provider adapter.

    Configuration supplied explicitly takes precedence over
    configuration from PROVIDER_REGISTRY.
    """

    provider_name = provider_name.strip().lower()

    provider_class = PROVIDER_CLASSES.get(
        provider_name
    )

    if provider_class is None:
        raise ValueError(
            f"Unsupported provider: {provider_name}"
        )

    registry_config = PROVIDER_REGISTRY.get(
        provider_name
    )

    if registry_config is None:
        raise ValueError(
            f"Provider is not registered: {provider_name}"
        )

    if config is None:

        config = {
            "api_key": registry_config.api_key,
            "base_url": registry_config.base_url,
            "account_id": registry_config.account_id,
            "timeout": registry_config.timeout,
            **registry_config.metadata,
        }

    return provider_class(
        config=config
    )


# ============================================================
# SAFE FACTORY
# ============================================================

def get_provider(
    provider_name: str,
) -> Optional[BaseProvider]:
    """
    Create a provider when it is registered.

    Returns None for unknown providers.
    """

    provider_name = provider_name.strip().lower()

    if provider_name not in PROVIDER_REGISTRY:
        return None

    return create_provider(
        provider_name
    )


# ============================================================
# ALL PROVIDERS
# ============================================================

def create_all_providers() -> dict[str, BaseProvider]:
    """
    Create every registered provider.

    Providers that cannot be instantiated are skipped.
    """

    result: dict[str, BaseProvider] = {}

    for provider_name in PROVIDER_REGISTRY:

        try:
            provider = create_provider(
                provider_name
            )

        except (ValueError, TypeError):
            continue

        result[provider_name] = provider

    return result


# ============================================================
# GLOBAL PROVIDER INSTANCES
# ============================================================

PROVIDERS = create_all_providers()