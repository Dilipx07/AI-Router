"""
Cloudflare Workers AI provider adapter for AI Router v1.

Implements the BaseProvider interface using Cloudflare's
Workers AI REST API.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .base import (
    BaseProvider,
    ProviderCapabilities,
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)


class CloudflareProvider(BaseProvider):
    """
    Cloudflare Workers AI provider.

    Configuration:
        api_key: Cloudflare API token
        account_id: Cloudflare account ID
        base_url: Optional API base URL
        timeout: Request timeout in seconds
    """

    name = "cloudflare"

    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        tools=True,
        structured_output=True,
        vision=False,
        embeddings=True,
        model_discovery=False,
    )

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)

        self.api_key = str(
            self.config.get("api_key", "")
        ).strip()

        self.account_id = str(
            self.config.get("account_id", "")
        ).strip()

        self.base_url = str(
            self.config.get(
                "base_url",
                "https://api.cloudflare.com/client/v4/accounts",
            )
        ).rstrip("/")

        self.timeout = float(
            self.config.get("timeout", 90)
        )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(self) -> bool:
        """
        Return True when Cloudflare credentials are configured.
        """

        return bool(
            self.api_key
            and self.account_id
        )

    # ========================================================
    # ENDPOINT
    # ========================================================

    def _endpoint(
        self,
        model: str,
    ) -> str:
        """
        Build the Workers AI run endpoint.
        """

        return (
            f"{self.base_url}/"
            f"{self.account_id}/ai/run/"
            f"{model}"
        )

    # ========================================================
    # HEADERS
    # ========================================================

    def _headers(self) -> dict[str, str]:
        """
        Build Cloudflare API headers.
        """

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ========================================================
    # REQUEST PAYLOAD
    # ========================================================

    @staticmethod
    def _build_payload(
        request: ProviderRequest,
    ) -> dict[str, Any]:
        """
        Convert ProviderRequest into a Workers AI payload.
        """

        payload: dict[str, Any] = {
            "messages": request.messages,
        }

        if request.tools:
            payload["tools"] = request.tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        return payload

    # ========================================================
    # RESPONSE CONTENT
    # ========================================================

    @staticmethod
    def _extract_content(
        data: dict[str, Any],
    ) -> str:
        """
        Extract textual content from common Workers AI
        response formats.
        """

        result = data.get("result")

        if isinstance(result, dict):

            response = result.get("response")

            if isinstance(response, str):
                return response

            if isinstance(response, dict):

                content = response.get("content")

                if isinstance(content, str):
                    return content

            content = result.get("content")

            if isinstance(content, str):
                return content

        response = data.get("response")

        if isinstance(response, str):
            return response

        if isinstance(response, dict):

            content = response.get("content")

            if isinstance(content, str):
                return content

        content = data.get("content")

        if isinstance(content, str):
            return content

        return ""

    # ========================================================
    # TOOL CALLS
    # ========================================================

    @staticmethod
    def _extract_tool_calls(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract tool calls from common response structures.
        """

        result = data.get("result")

        if not isinstance(result, dict):
            result = {}

        response = result.get("response")

        if not isinstance(response, dict):
            response = {}

        tool_calls = response.get("tool_calls")

        if isinstance(tool_calls, list):
            return [
                item
                for item in tool_calls
                if isinstance(item, dict)
            ]

        tool_calls = result.get("tool_calls")

        if isinstance(tool_calls, list):
            return [
                item
                for item in tool_calls
                if isinstance(item, dict)
            ]

        tool_calls = data.get("tool_calls")

        if isinstance(tool_calls, list):
            return [
                item
                for item in tool_calls
                if isinstance(item, dict)
            ]

        return []

    # ========================================================
    # FINISH REASON
    # ========================================================

    @staticmethod
    def _extract_finish_reason(
        data: dict[str, Any],
    ) -> str | None:
        """
        Extract finish reason when available.
        """

        result = data.get("result")

        if isinstance(result, dict):

            response = result.get("response")

            if isinstance(response, dict):

                value = response.get(
                    "finish_reason"
                )

                if isinstance(value, str):
                    return value

            value = result.get("finish_reason")

            if isinstance(value, str):
                return value

        value = data.get("finish_reason")

        if isinstance(value, str):
            return value

        return None

    # ========================================================
    # USAGE
    # ========================================================

    @staticmethod
    def _extract_usage(
        data: dict[str, Any],
    ) -> ProviderUsage:
        """
        Extract token usage from Cloudflare Workers AI
        response structures.

        Cloudflare response schemas can vary by model/provider
        version, so common locations are checked.
        """

        candidates: list[Any] = [
            data.get("usage"),
        ]

        result = data.get("result")

        if isinstance(result, dict):

            candidates.extend(
                [
                    result.get("usage"),
                ]
            )

            response = result.get("response")

            if isinstance(response, dict):
                candidates.append(
                    response.get("usage")
                )

        for usage in candidates:

            if not isinstance(usage, dict):
                continue

            prompt_tokens = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("promptTokenCount")
                or 0
            )

            completion_tokens = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("candidatesTokenCount")
                or 0
            )

            total_tokens = (
                usage.get("total_tokens")
                or usage.get("totalTokenCount")
                or 0
            )

            return ProviderUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ).normalized()

        return ProviderUsage()

    # ========================================================
    # CHAT
    # ========================================================

    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a chat request against Cloudflare Workers AI.
        """

        if not self.is_available():

            return ProviderResult.failure_result(
                provider=self.provider_name,
                model=model,
                status="auth_failure",
                error=(
                    "Cloudflare provider is not configured. "
                    "Both api_key and account_id are required."
                ),
            )

        started = time.perf_counter()

        url = self._endpoint(model)

        payload = self._build_payload(request)

        try:

            response = httpx.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            try:
                data = response.json()
                usage = self._extract_usage(data)
            except ValueError:
                data = {}

            if response.status_code >= 400:

                error_message = (
                    self._extract_error(data)
                    or response.text
                    or f"HTTP {response.status_code}"
                )

                return ProviderResult.failure_result(
                    provider=self.provider_name,
                    model=model,
                    status="error",
                    error=error_message,
                    elapsed=elapsed,
                    http_status=response.status_code,
                    raw_response=data,
                    usage=usage,
                )

            content = self._extract_content(
                data
            )

            tool_calls = self._extract_tool_calls(
                data
            )

            finish_reason = (
                self._extract_finish_reason(
                    data
                )
            )

            if not content and not tool_calls:

                return ProviderResult.failure_result(
                    provider=self.provider_name,
                    model=model,
                    status="empty_response",
                    error="Cloudflare returned an empty response.",
                    elapsed=elapsed,
                    http_status=response.status_code,
                    raw_response=data,
                )

            return ProviderResult.success_result(
                provider=self.provider_name,
                model=model,
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                elapsed=elapsed,
                http_status=response.status_code,
                raw_response=data,
            )

        except httpx.TimeoutException as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            return ProviderResult.failure_result(
                provider=self.provider_name,
                model=model,
                status="timeout",
                error=str(exc) or "Cloudflare request timed out.",
                elapsed=elapsed,
            )

        except httpx.HTTPError as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            return ProviderResult.failure_result(
                provider=self.provider_name,
                model=model,
                status="network_failure",
                error=str(exc),
                elapsed=elapsed,
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - started
            )

            return ProviderResult.failure_result(
                provider=self.provider_name,
                model=model,
                status="provider_failure",
                error=str(exc),
                elapsed=elapsed,
            )

    # ========================================================
    # ERROR EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_error(
        data: dict[str, Any],
    ) -> str | None:
        """
        Extract an error message from Cloudflare responses.
        """

        errors = data.get("errors")

        if isinstance(errors, list):

            messages = []

            for error in errors:

                if isinstance(error, dict):

                    message = error.get("message")

                    if isinstance(message, str):
                        messages.append(message)

                elif isinstance(error, str):
                    messages.append(error)

            if messages:
                return "; ".join(messages)

        error = data.get("error")

        if isinstance(error, str):
            return error

        if isinstance(error, dict):

            message = error.get("message")

            if isinstance(message, str):
                return message

        return None