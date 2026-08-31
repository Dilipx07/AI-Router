"""
AI Router v1
OpenAI-compatible provider implementation.

Used as the common transport layer for providers exposing
OpenAI-compatible chat completion APIs.

Examples:
- Groq
- OpenRouter
- Hugging Face
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from .base import (
    BaseProvider,
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)
from router.errors import (
    ErrorType,
    RouterError,
    error_from_http_status,
)


# ============================================================
# OPENAI-COMPATIBLE PROVIDER
# ============================================================

class OpenAICompatibleProvider(BaseProvider):
    """
    Base implementation for OpenAI-compatible APIs.

    Provider-specific classes should primarily configure:
    - name
    - API key
    - base URL
    - timeout
    """

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:

        super().__init__(config=config)

        self.api_key = self.config.get(
            "api_key",
            "",
        )

        self.base_url = (
            self.config.get(
                "base_url",
                "",
            )
            .rstrip("/")
        )

        self.timeout = float(
            self.config.get(
                "timeout",
                30,
            )
        )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(self) -> bool:
        """
        Return whether the provider has enough configuration
        to attempt an API request.
        """

        return bool(
            self.api_key
            and self.base_url
        )

    # ========================================================
    # USAGE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_usage(
        response_json: dict[str, Any],
    ) -> ProviderUsage:
        """
        Extract token usage from an OpenAI-compatible response.
        """

        usage = response_json.get(
            "usage",
            {},
        )

        if not isinstance(
            usage,
            dict,
        ):
            return ProviderUsage()

        return ProviderUsage(
            prompt_tokens=usage.get(
                "prompt_tokens",
                0,
            ) or 0,

            completion_tokens=usage.get(
                "completion_tokens",
                0,
            ) or 0,

            total_tokens=usage.get(
                "total_tokens",
                0,
            ) or 0,
        ).normalized()

    # ========================================================
    # HEADERS
    # ========================================================

    def _headers(self) -> dict[str, str]:
        """
        Build standard OpenAI-compatible request headers.
        """

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ========================================================
    # REQUEST PAYLOAD
    # ========================================================

    @staticmethod
    def _build_payload(
        *,
        model: str,
        request: ProviderRequest,
    ) -> dict[str, Any]:
        """
        Convert ProviderRequest into an OpenAI-compatible
        chat completion payload.
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
        }

        if request.tools is not None:
            payload["tools"] = request.tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature
            
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        return payload

    # ========================================================
    # RESPONSE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_content(
        response_json: dict[str, Any],
    ) -> str:
        """
        Extract textual content from an OpenAI-compatible
        response.
        """

        choices = response_json.get(
            "choices",
            [],
        )

        if not choices:
            return ""

        message = choices[0].get(
            "message",
            {},
        )

        content = message.get(
            "content",
            "",
        )

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # Some providers may return structured content.
        if isinstance(content, list):

            text_parts: list[str] = []

            for item in content:

                if not isinstance(item, dict):
                    continue

                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

            return "".join(text_parts)

        return str(content)

    # ========================================================
    # TOOL CALL EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_tool_calls(
        response_json: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract normalized tool calls.

        We intentionally preserve the provider's tool-call
        structure for now. A deeper normalization layer can
        be added later.
        """

        choices = response_json.get(
            "choices",
            [],
        )

        if not choices:
            return []

        message = choices[0].get(
            "message",
            {},
        )

        tool_calls = message.get(
            "tool_calls",
            [],
        )

        if not isinstance(tool_calls, list):
            return []

        return [
            call
            for call in tool_calls
            if isinstance(call, dict)
        ]

    # ========================================================
    # FINISH REASON
    # ========================================================

    @staticmethod
    def _extract_finish_reason(
        response_json: dict[str, Any],
    ) -> Optional[str]:

        choices = response_json.get(
            "choices",
            [],
        )

        if not choices:
            return None

        reason = choices[0].get(
            "finish_reason"
        )

        if reason is None:
            return None

        return str(reason)

    # ========================================================
    # ERROR MESSAGE
    # ========================================================

    @staticmethod
    def _extract_error_message(
        response_json: Any,
    ) -> str:
        """
        Extract a useful error message from common API formats.
        """

        if not isinstance(
            response_json,
            dict,
        ):
            return ""

        error = response_json.get(
            "error"
        )

        if isinstance(error, dict):

            message = error.get(
                "message"
            )

            if message:
                return str(message)

            return str(error)

        if isinstance(error, str):
            return error

        message = response_json.get(
            "message"
        )

        if message:
            return str(message)

        return ""

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a synchronous OpenAI-compatible chat request.

        Provider-specific API errors are converted into
        RouterError so the execution layer can make routing
        decisions without knowing provider-specific semantics.
        """

        if not self.is_available():

            error = RouterError(
                ErrorType.AUTH_FAILURE,
                "Provider is not configured.",
                provider=self.provider_name,
                model=model,
                retryable=False,
            )

            raise error

        payload = self._build_payload(
            model=model,
            request=request,
        )

        started = time.perf_counter()

        try:

            with httpx.Client(
                timeout=self.timeout,
            ) as client:

                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )

            elapsed = (
                time.perf_counter()
                - started
            )

        except httpx.TimeoutException as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            raise RouterError(
                ErrorType.TIMEOUT,
                str(exc) or "Provider request timed out.",
                provider=self.provider_name,
                model=model,
                retryable=True,
                metadata={
                    "elapsed": elapsed,
                },
            ) from exc

        except httpx.RequestError as exc:
            elapsed = (time.perf_counter() - started)
            raise RouterError(
                ErrorType.NETWORK_FAILURE,
                str(exc) or "Provider network failure.",
                provider=self.provider_name,
                model=model,
                retryable=True,
                metadata={
                    "elapsed": elapsed,
                },
            ) from exc

        # ----------------------------------------------------
        # HTTP STATUS
        # ----------------------------------------------------

        try:
            response_json = response.json()
            usage = self._extract_usage(
                response_json
            )

        except ValueError:
            response_json = {}

        if response.status_code >= 400:
            message = self._extract_error_message(
                response_json
            )
            raise error_from_http_status(
                status_code=response.status_code,
                provider=self.provider_name,
                model=model,
                message=(
                    message
                    or response.text[:500]
                ),
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        content = self._extract_content(
            response_json
        )
        tool_calls = self._extract_tool_calls(
            response_json
        )
        finish_reason = self._extract_finish_reason(
            response_json
        )
        if not content.strip() and not tool_calls:
            raise RouterError(
                ErrorType.EMPTY_RESPONSE,
                "Provider returned an empty response.",
                provider=self.provider_name,
                model=model,
                retryable=True,
                metadata={
                    "elapsed": elapsed,
                },
            )

        return ProviderResult.success_result(
            provider=self.provider_name,
            model=model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            elapsed=elapsed,
            http_status=response.status_code,
            raw_response=response_json,
            usage=usage,
        )