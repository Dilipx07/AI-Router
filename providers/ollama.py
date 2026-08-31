"""
AI Router v1
Ollama provider adapter.

Ollama exposes an OpenAI-compatible API. This adapter reuses
the shared OpenAI-compatible transport while keeping Ollama-
specific configuration isolated from the routing engine.

Supports:
- local Ollama models
- Ollama cloud models
- tools
- streaming capability declaration
- structured output capability declaration
- vision-capable model metadata when configured
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ProviderCapabilities, ProviderRequest, ProviderResult
from .openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """
    Ollama provider implementation.

    The router sees Ollama through the standard provider
    abstraction and does not need to know whether the selected
    model is local or a cloud model.
    """

    name = "ollama"

    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        structured_output=True,
        vision=False,
        embeddings=False,
        model_discovery=True,
    )

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)

        self.base_url = (
            self.config.get(
                "base_url",
                "http://localhost:11434",
            )
            .rstrip("/")
        )

        # Ollama's OpenAI-compatible API lives under /v1.
        self.api_base_url = f"{self.base_url}/v1"

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
                120,
            )
        )

    # --------------------------------------------------------
    # AVAILABILITY
    # --------------------------------------------------------

    def is_available(self) -> bool:
        """
        Ollama does not require an API key.

        A configured base URL is sufficient for the router to
        consider the provider eligible. Actual connectivity
        failures are handled by the execution layer.
        """

        return bool(
            self.base_url
        )

    # --------------------------------------------------------
    # HEADERS
    # --------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """
        Build Ollama-compatible request headers.

        Ollama does not require Authorization for normal local
        deployments. If an API key is configured, preserve it
        for deployments that support authenticated access.
        """

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a chat completion through Ollama's
        OpenAI-compatible /v1 endpoint.
        """

        original_base_url = self.base_url

        try:
            self.base_url = f"{original_base_url}/v1"

            return super().chat(
                model=model,
                request=request,
            )

        finally:
            self.base_url = original_base_url

    # --------------------------------------------------------
    # MODEL DISCOVERY
    # --------------------------------------------------------

    def discover_models(self) -> list[str]:
        """
        Return models currently exposed by Ollama.

        The shared OpenAI-compatible transport is intentionally
        not used here because Ollama's model discovery endpoint
        is provider-specific.
        """

        import httpx

        if not self.is_available():
            return []

        try:
            with httpx.Client(
                timeout=self.timeout,
            ) as client:
                response = client.get(
                    f"{self.base_url}/api/tags",
                    headers=self._headers(),
                )

            if response.status_code >= 400:
                return []

            data = response.json()

        except (
            httpx.RequestError,
            ValueError,
        ):
            return []

        models = data.get(
            "models",
            [],
        )

        if not isinstance(models, list):
            return []

        result: list[str] = []

        for item in models:
            if not isinstance(item, dict):
                continue

            name = item.get("name")

            if isinstance(name, str) and name:
                result.append(name)

        return result

    # --------------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.provider_name!r})"
        )