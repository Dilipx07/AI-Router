"""
AI Router v1
Groq provider adapter.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import ProviderCapabilities
from .openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """
    Groq provider using the OpenAI-compatible API.
    """

    name = "groq"

    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        structured_output=True,
        vision=False,
        embeddings=False,
        model_discovery=False,
    )

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:

        super().__init__(config=config)