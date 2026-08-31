"""
Central provider and model registry for AI Router v1.

This module contains configuration metadata only.

Provider implementations belong in providers/.
Routing logic belongs in router/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# ENVIRONMENT
# ============================================================

def env(name: str, default: str = "") -> str:
    """Read an environment variable and strip whitespace."""
    return os.getenv(name, default).strip()


# ============================================================
# MODEL DEFINITION
# ============================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a single model.
    """

    name: str

    priority: int = 100

    enabled: bool = True

    capabilities: frozenset[str] = field(
        default_factory=frozenset
    )

    max_tokens: Optional[int] = None

    metadata: dict = field(
        default_factory=dict
    )


# ============================================================
# PROVIDER DEFINITION
# ============================================================

@dataclass
class ProviderConfig:
    """
    Configuration for an AI provider.
    """

    name: str

    enabled: bool = True

    api_key: str = ""

    base_url: str = ""

    account_id: str = ""

    timeout: int = 30

    models: list[ModelConfig] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )

    @property
    def available(self) -> bool:
        """
        Determine whether the provider has the credentials
        required to attempt requests.
        """

        if not self.enabled:
            return False

        # Ollama is local and does not require an API key.
        if self.name == "ollama":
            return True

        if self.name == "cloudflare":
            return bool(
                self.api_key
                and self.account_id
            )

        return bool(self.api_key)

    def get_models(self) -> list[ModelConfig]:
        """Return enabled models ordered by priority."""

        return sorted(
            (
                model
                for model in self.models
                if model.enabled
            ),
            key=lambda model: model.priority,
        )


# ============================================================
# MODEL HELPERS
# ============================================================

def model(
    name: str,
    priority: int,
    *capabilities: str,
) -> ModelConfig:
    """
    Convenience constructor for model definitions.
    """

    return ModelConfig(
        name=name,
        priority=priority,
        capabilities=frozenset(
            capabilities
        ),
    )


# ============================================================
# PROVIDER REGISTRY
# ============================================================

PROVIDER_REGISTRY: dict[str, ProviderConfig] = {

    # ========================================================
    # GOOGLE
    # ========================================================

    "google": ProviderConfig(
        name="google",

        api_key=env("GOOGLE_API_KEY"),

        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/models"
        ),

        models=[
            model(
                "gemini-3.1-flash-lite",
                10,
                "chat",
                "tools",
            ),
            model(
                "gemini-3.6-flash",
                20,
                "chat",
                "tools",
            ),
            model(
                "gemini-3.5-flash-lite",
                30,
                "chat",
                "tools",
            ),
            model(
                "gemini-3.5-flash",
                40,
                "chat",
                "tools",
            ),
            model(
                "gemini-2.5-flash",
                50,
                "chat",
                "tools",
            ),
            model(
                "gemini-2.5-flash-lite",
                60,
                "chat",
                "tools",
            ),
        ],
    ),

    # ========================================================
    # GROQ
    # ========================================================

    "groq": ProviderConfig(
        name="groq",

        api_key=env("GROQ_API_KEY"),

        base_url=(
            "https://api.groq.com/openai/v1"
        ),

        models=[
            model(
                "openai/gpt-oss-120b",
                10,
                "chat",
                "tools",
            ),
            model(
                "openai/gpt-oss-20b",
                20,
                "chat",
                "tools",
            ),
            model(
                "qwen/qwen3.6-27b",
                30,
                "chat",
                "tools",
            ),
        ],
    ),

    # ========================================================
    # OPENROUTER
    # ========================================================

    "openrouter": ProviderConfig(
        name="openrouter",

        api_key=env("OPENROUTER_API_KEY"),

        base_url=(
            "https://openrouter.ai/api/v1"
        ),

        models=[
            model(
                "openai/gpt-oss-20b:free",
                10,
                "chat",
                "tools",
            ),
            model(
                "nvidia/nemotron-3.5-lightning:free",
                20,
                "chat",
                "tools",
            ),
            model(
                "cohere/north-mini-code:free",
                30,
                "chat",
                "tools",
            ),
            model(
                "z-ai/glm-5.2:free",
                40,
                "chat",
                "tools",
            ),
        ],
    ),

    # ========================================================
    # CLOUDFLARE
    # ========================================================

    "cloudflare": ProviderConfig(
        name="cloudflare",

        api_key=env("CLOUDFLARE_API_KEY"),

        account_id=env(
            "CLOUDFLARE_ACCOUNT_ID"
        ),

        base_url=(
            "https://api.cloudflare.com/"
            "client/v4/accounts"
        ),

        models=[
            model(
                "@cf/qwen/qwen3.8-27b",
                10,
                "chat",
                "tools",
            ),
            model(
                "@cf/zai-org/glm-5.2",
                20,
                "chat",
                "tools",
            ),
            model(
                "@cf/moonshotai/kimi-k2.7-code",
                30,
                "chat",
                "tools",
            ),
            model(
                "@cf/deepseek-ai/deepseek-v4-flash-0731",
                40,
                "chat",
                "tools",
            ),
            model(
                "@cf/deepseek-ai/deepseek-v4-pro-0813",
                50,
                "chat",
                "tools",
            ),
            model(
                "@cf/openai/gpt-oss-120b",
                60,
                "chat",
                "tools",
            ),
            model(
                "@cf/openai/gpt-oss-20b",
                70,
                "chat",
                "tools",
            ),
            model(
                "@cf/qwen/qwen2.5-coder-32b-instruct",
                80,
                "chat",
                "tools",
            ),
            model(
                "@cf/qwen/qwen3-30b-a3b-fp8",
                90,
                "chat",
                "tools",
            ),
            model(
                "@cf/zai-org/glm-4.7-flash",
                100,
                "chat",
                "tools",
            ),
            model(
                "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                110,
                "chat",
                "tools",
            ),
            model(
                "@cf/meta/llama-3.1-8b-instruct-fp8",
                120,
                "chat",
                "tools",
            ),
        ],
    ),

    # ========================================================
    # HUGGING FACE
    # ========================================================

    "huggingface": ProviderConfig(
        name="huggingface",

        api_key=env("HF_API_KEY"),

        base_url=(
            "https://router.huggingface.co/v1"
        ),

        timeout=90,

        models=[
            model(
                "Qwen/Qwen3-Coder-Next",
                10,
                "chat",
                "tools",
            ),
            model(
                "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                20,
                "chat",
                "tools",
            ),
            model(
                "deepseek-ai/DeepSeek-V4-Flash",
                30,
                "chat",
                "tools",
            ),
            model(
                "moonshotai/Kimi-K2.7-Code",
                40,
                "chat",
                "tools",
            ),
            model(
                "zai-org/GLM-5.2",
                50,
                "chat",
                "tools",
            ),
            model(
                "MiniMaxAI/MiniMax-M3",
                60,
                "chat",
                "tools",
            ),
            model(
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                70,
                "chat",
                "tools",
            ),
            model(
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                80,
                "chat",
                "tools",
            ),
        ],
    ),

    # ========================================================
    # OLLAMA
    # ========================================================

    "ollama": ProviderConfig(
        name="ollama",

        base_url=env(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),

        timeout=120,

        models=[
            model(
                "minimax-m3:cloud",
                10,
                "chat",
                "tools",
            ),
            model(
                "gpt-oss:120b-cloud",
                20,
                "chat",
                "tools",
            ),
            model(
                "nemotron-3-ultra:cloud",
                30,
                "chat",
                "tools",
            ),
            model(
                "gpt-oss:20b-cloud",
                40,
                "chat",
                "tools",
            ),
            model(
                "nemotron-3-super:cloud",
                50,
                "chat",
                "tools",
            ),
            model(
                "nemotron-3-nano:30b-cloud",
                60,
                "chat",
                "tools",
            ),
            model(
                "gemma4:31b-cloud",
                70,
                "chat",
                "tools",
            ),
        ],
    ),
}


# ============================================================
# PROVIDER ORDER
# ============================================================

PROVIDER_ORDER: tuple[str, ...] = (
    "google",
    "groq",
    "openrouter",
    "cloudflare",
    "huggingface",
    "ollama",
)


# ============================================================
# REGISTRY ACCESS
# ============================================================

def get_provider(
    provider_name: str,
) -> Optional[ProviderConfig]:
    """Return a provider configuration by name."""

    return PROVIDER_REGISTRY.get(
        provider_name
    )


def get_enabled_providers() -> list[ProviderConfig]:
    """Return enabled providers in routing order."""

    result = []

    for name in PROVIDER_ORDER:

        provider = PROVIDER_REGISTRY.get(name)

        if provider and provider.enabled:
            result.append(provider)

    return result


def get_configured_providers() -> list[ProviderConfig]:
    """Return providers that currently have credentials."""

    return [
        provider
        for provider in get_enabled_providers()
        if provider.available
    ]