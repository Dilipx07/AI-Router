"""
AI Router v1
Hugging Face provider adapter.

Hugging Face Inference Providers exposes an OpenAI-compatible
chat completion endpoint, so this provider reuses the shared
OpenAI-compatible transport layer.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ProviderCapabilities, ProviderRequest, ProviderResult
from .openai_compatible import OpenAICompatibleProvider


class HuggingFaceProvider(OpenAICompatibleProvider):
    """
    Hugging Face provider implementation.

    Provider-specific configuration stays inside this adapter.
    The routing engine only sees the normalized provider
    interface.
    """

    name = "huggingface"

    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        structured_output=True,
        vision=True,
        embeddings=False,
        model_discovery=False,
    )

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)

        self.base_url = (
            self.config.get(
                "base_url",
                "https://router.huggingface.co/v1",
            )
            .rstrip("/")
        )

        self.api_key = (
            self.config.get(
                "api_key",
                "",
            )
            or ""
        ).strip()

        self.timeout = float(
            self.config.get(
                "timeout",
                90,
            )
        )

    def is_available(self) -> bool:
        """
        Return True when Hugging Face is configured.
        """

        return bool(
            self.api_key
            and self.base_url
        )

    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a chat completion through the shared
        OpenAI-compatible transport.
        """

        return super().chat(
            model=model,
            request=request,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.provider_name!r})"
        )