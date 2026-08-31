"""
AI Router v1
OpenRouter provider adapter.

OpenRouter exposes an OpenAI-compatible API, so this provider
inherits the shared OpenAI-compatible implementation and adds
OpenRouter-specific identity and request metadata.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResult,
)
from .openai_compatible import OpenAICompatibleProvider


# ============================================================
# PROVIDER
# ============================================================


class OpenRouterProvider(OpenAICompatibleProvider):
    """
    OpenRouter provider implementation.

    OpenRouter is treated as an OpenAI-compatible provider.

    Provider-specific API details remain inside this adapter.
    The routing engine does not need to know that OpenRouter
    is being used.
    """

    name = "openrouter"

    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        structured_output=True,
        vision=True,
        embeddings=False,
        model_discovery=False,
    )

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            config=config,
        )

        self.base_url = (
            self.config.get(
                "base_url",
                "https://openrouter.ai/api/v1",
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

        self.timeout = int(
            self.config.get(
                "timeout",
                30,
            )
        )

        self.http_referer = (
            self.config.get(
                "http_referer",
                "",
            )
            or ""
        ).strip()

        self.app_title = (
            self.config.get(
                "app_title",
                "AI Router v1",
            )
            or "AI Router v1"
        ).strip()

    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    def is_available(self) -> bool:
        """
        Return True when an OpenRouter API key is configured.
        """

        return bool(
            self.api_key
        )

    # --------------------------------------------------------
    # HEADERS
    # --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """
        Build OpenRouter request headers.

        Authorization is required.

        HTTP-Referer and X-Title are optional OpenRouter
        metadata headers and are only sent when configured.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.http_referer:
            headers["HTTP-Referer"] = (
                self.http_referer
            )

        if self.app_title:
            headers["X-Title"] = (
                self.app_title
            )

        return headers

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a chat completion through OpenRouter.

        The actual HTTP implementation is delegated to the
        shared OpenAI-compatible provider.
        """

        # OpenRouter uses the normal OpenAI-compatible
        # /chat/completions endpoint.
        return super().chat(
            model=model,
            request=request,
        )

    # --------------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.provider_name!r})"
        )