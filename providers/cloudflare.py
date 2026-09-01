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
    # MESSAGE NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_message_content(
        content: Any,
    ) -> str | None:
        """
        Normalize OpenAI/Cline message content into the
        string/null format expected by Cloudflare Workers AI.
        """

        if content is None:
            return None

        if isinstance(content, str):
            return content

        if isinstance(content, list):

            parts: list[str] = []

            for item in content:

                if isinstance(item, str):
                    parts.append(item)
                    continue

                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")

                # --------------------------------------------
                # TEXT
                # --------------------------------------------

                if item_type == "text":

                    text = item.get("text")

                    if isinstance(text, str):
                        parts.append(text)

                    continue

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                if item_type == "image_url":

                    image_url = item.get(
                        "image_url"
                    )

                    if isinstance(
                        image_url,
                        dict,
                    ):
                        url = image_url.get("url")
                    else:
                        url = image_url

                    if isinstance(url, str):
                        parts.append(
                            f"[image: {url}]"
                        )

                    continue

                # --------------------------------------------
                # GENERIC TEXT
                # --------------------------------------------

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)
                    continue

                nested_content = item.get(
                    "content"
                )

                if isinstance(
                    nested_content,
                    str,
                ):
                    parts.append(
                        nested_content
                    )

            return "\n".join(
                part
                for part in parts
                if part
            )

        return str(content)

    @classmethod
    def _normalize_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert incoming OpenAI/Cline messages into the
        schema accepted by Cloudflare Workers AI.
        """

        normalized: list[dict[str, Any]] = []

        for message in messages:

            if not isinstance(message, dict):
                continue

            role = message.get("role")

            if not isinstance(
                role,
                str,
            ):
                continue

            if not role:
                continue

            content = cls._normalize_message_content(
                message.get("content")
            )

            # ------------------------------------------------
            # TOOL RESULT
            # ------------------------------------------------

            if role == "tool":

                tool_name = message.get(
                    "name"
                )

                tool_call_id = message.get(
                    "tool_call_id"
                )

                prefix = "[tool result]"

                if isinstance(
                    tool_name,
                    str,
                ) and tool_name:
                    prefix += (
                        f" {tool_name}"
                    )

                if isinstance(
                    tool_call_id,
                    str,
                ) and tool_call_id:
                    prefix += (
                        f" ({tool_call_id})"
                    )

                if content is None:
                    content = ""

                normalized.append(
                    {
                        "role": "user",
                        "content": (
                            f"{prefix}\n"
                            f"{content}"
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # ASSISTANT TOOL CALLS
            # ------------------------------------------------

            if role == "assistant":

                tool_calls = message.get(
                    "tool_calls"
                )

                if isinstance(
                    tool_calls,
                    list,
                ) and tool_calls:

                    tool_lines: list[str] = []

                    for tool_call in tool_calls:

                        if not isinstance(
                            tool_call,
                            dict,
                        ):
                            continue

                        function = tool_call.get(
                            "function",
                            {},
                        )

                        if not isinstance(
                            function,
                            dict,
                        ):
                            function = {}

                        name = function.get(
                            "name",
                            "",
                        )

                        arguments = function.get(
                            "arguments",
                            "",
                        )

                        if not isinstance(
                            name,
                            str,
                        ):
                            name = str(name)

                        if not isinstance(
                            arguments,
                            str,
                        ):
                            arguments = str(
                                arguments
                            )

                        tool_lines.append(
                            (
                                "[tool call] "
                                f"{name}"
                                f"({arguments})"
                            )
                        )

                    if tool_lines:

                        tool_text = "\n".join(
                            tool_lines
                        )

                        if content:

                            content = (
                                f"{content}\n"
                                f"{tool_text}"
                            )

                        else:

                            content = tool_text

                if content is None:
                    content = ""

                normalized.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )

                continue

            # ------------------------------------------------
            # USER / SYSTEM
            # ------------------------------------------------

            if content is None:
                content = ""

            normalized.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized

    # ========================================================
    # TOOL SCHEMA NORMALIZATION
    # ========================================================

    @classmethod
    def _sanitize_tool_schema(
        cls,
        schema: Any,
    ) -> Any:
        """
        Convert arbitrary OpenAI/JSON-Schema definitions into
        a provider-compatible schema.

        Cline can send full JSON Schema containing fields such as:

            $schema
            additionalProperties
            exclusiveMinimum
            exclusiveMaximum
            $ref
            oneOf
            anyOf
            nullable
            default
            examples

        Cloudflare's function-tool parameter schema is more
        restrictive, so unsupported JSON-Schema keywords are
        removed before the request is sent.
        """

        if isinstance(
            schema,
            list,
        ):

            return [
                cls._sanitize_tool_schema(
                    item
                )
                for item in schema
            ]

        if not isinstance(
            schema,
            dict,
        ):
            return schema

        # Keep only fields useful for function parameter schemas.
        allowed_keys = {
            "type",
            "description",
            "properties",
            "required",
            "items",
            "enum",
            "format",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "pattern",
            "minItems",
            "maxItems",
        }

        result: dict[str, Any] = {}

        for key, value in schema.items():

            if key not in allowed_keys:
                continue

            result[key] = cls._sanitize_tool_schema(
                value
            )

        # ----------------------------------------------------
        # NORMALIZE NULLABLE UNION TYPES
        # ----------------------------------------------------

        for union_key in (
            "anyOf",
            "oneOf",
        ):

            if union_key not in schema:
                continue

            union_values = schema.get(
                union_key
            )

            if not isinstance(
                union_values,
                list,
            ):
                continue

            usable = [
                item
                for item in union_values
                if (
                    isinstance(
                        item,
                        dict,
                    )
                    and item.get("type")
                    != "null"
                )
            ]

            if len(usable) == 1:

                normalized = cls._sanitize_tool_schema(
                    usable[0]
                )

                if isinstance(
                    normalized,
                    dict,
                ):

                    result.update(
                        normalized
                    )

            break

        # ----------------------------------------------------
        # JSON SCHEMA CONST
        # ----------------------------------------------------

        if "const" in schema:

            value = schema.get(
                "const"
            )

            if "enum" not in result:
                result["enum"] = [value]

        # ----------------------------------------------------
        # ENSURE OBJECT PROPERTY SCHEMAS ARE DICTS
        # ----------------------------------------------------

        properties = result.get(
            "properties"
        )

        if isinstance(
            properties,
            dict,
        ):

            cleaned_properties: dict[
                str,
                Any,
            ] = {}

            for name, value in properties.items():

                if not isinstance(
                    name,
                    str,
                ):
                    continue

                cleaned = cls._sanitize_tool_schema(
                    value
                )

                if isinstance(
                    cleaned,
                    dict,
                ):
                    cleaned_properties[name] = (
                        cleaned
                    )

            result["properties"] = (
                cleaned_properties
            )

        # ----------------------------------------------------
        # REQUIRED
        # ----------------------------------------------------

        required = result.get(
            "required"
        )

        if isinstance(
            required,
            list,
        ):

            result["required"] = [
                name
                for name in required
                if isinstance(
                    name,
                    str,
                )
            ]

        return result

    @classmethod
    def _normalize_tools(
        cls,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Convert OpenAI function tools into Cloudflare-compatible
        function tool definitions.

        Cloudflare accepts the OpenAI-style wrapper:

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """

        if not tools:
            return []

        normalized: list[dict[str, Any]] = []

        for tool in tools:

            if not isinstance(
                tool,
                dict,
            ):
                continue

            function = tool.get(
                "function"
            )

            if not isinstance(
                function,
                dict,
            ):
                continue

            name = function.get(
                "name"
            )

            if not isinstance(
                name,
                str,
            ) or not name:

                continue

            normalized_function: dict[
                str,
                Any,
            ] = {
                "name": name,
            }

            description = function.get(
                "description"
            )

            if isinstance(
                description,
                str,
            ):
                normalized_function[
                    "description"
                ] = description

            parameters = function.get(
                "parameters"
            )

            if isinstance(
                parameters,
                dict,
            ):

                normalized_function[
                    "parameters"
                ] = cls._sanitize_tool_schema(
                    parameters
                )

            normalized.append(
                {
                    "type": "function",
                    "function": normalized_function,
                }
            )

        return normalized

    # ========================================================
    # REQUEST PAYLOAD
    # ========================================================

    @classmethod
    def _build_payload(
        cls,
        request: ProviderRequest,
    ) -> dict[str, Any]:
        """
        Build the provider-specific Cloudflare request payload.
        """

        messages = cls._normalize_messages(
            request.messages
        )

        payload: dict[str, Any] = {
            "messages": messages,
        }

        tools = cls._normalize_tools(
            request.tools
        )

        if tools:
            payload["tools"] = tools

        if request.tool_choice is not None:

            payload["tool_choice"] = (
                request.tool_choice
            )

        if request.max_tokens is not None:

            payload["max_tokens"] = (
                request.max_tokens
            )

        if request.temperature is not None:

            payload["temperature"] = (
                request.temperature
            )

        return payload

    # ========================================================
    # RESPONSE CONTENT
    # ========================================================

    @staticmethod
    def _extract_content(
        data: dict[str, Any],
    ) -> str:

        result = data.get(
            "result"
        )

        if isinstance(
            result,
            dict,
        ):

            response = result.get(
                "response"
            )

            if isinstance(
                response,
                str,
            ):
                return response

            if isinstance(
                response,
                dict,
            ):

                content = response.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ):
                    return content

            content = result.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):
                return content

        response = data.get(
            "response"
        )

        if isinstance(
            response,
            str,
        ):
            return response

        if isinstance(
            response,
            dict,
        ):

            content = response.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):
                return content

        content = data.get(
            "content"
        )

        if isinstance(
            content,
            str,
        ):
            return content

        return ""

    # ========================================================
    # TOOL CALLS
    # ========================================================

    @staticmethod
    def _extract_tool_calls(
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:

        result = data.get(
            "result"
        )

        if not isinstance(
            result,
            dict,
        ):
            result = {}

        response = result.get(
            "response"
        )

        if not isinstance(
            response,
            dict,
        ):
            response = {}

        tool_calls = response.get(
            "tool_calls"
        )

        if isinstance(
            tool_calls,
            list,
        ):

            return [
                item
                for item in tool_calls
                if isinstance(
                    item,
                    dict,
                )
            ]

        tool_calls = result.get(
            "tool_calls"
        )

        if isinstance(
            tool_calls,
            list,
        ):

            return [
                item
                for item in tool_calls
                if isinstance(
                    item,
                    dict,
                )
            ]

        tool_calls = data.get(
            "tool_calls"
        )

        if isinstance(
            tool_calls,
            list,
        ):

            return [
                item
                for item in tool_calls
                if isinstance(
                    item,
                    dict,
                )
            ]

        return []

    # ========================================================
    # FINISH REASON
    # ========================================================

    @staticmethod
    def _extract_finish_reason(
        data: dict[str, Any],
    ) -> str | None:

        result = data.get(
            "result"
        )

        if isinstance(
            result,
            dict,
        ):

            response = result.get(
                "response"
            )

            if isinstance(
                response,
                dict,
            ):

                value = response.get(
                    "finish_reason"
                )

                if isinstance(
                    value,
                    str,
                ):
                    return value

            value = result.get(
                "finish_reason"
            )

            if isinstance(
                value,
                str,
            ):
                return value

        value = data.get(
            "finish_reason"
        )

        if isinstance(
            value,
            str,
        ):
            return value

        return None

    # ========================================================
    # USAGE
    # ========================================================

    @staticmethod
    def _extract_usage(
        data: dict[str, Any],
    ) -> ProviderUsage:

        candidates: list[Any] = [
            data.get("usage")
        ]

        result = data.get(
            "result"
        )

        if isinstance(
            result,
            dict,
        ):

            candidates.append(
                result.get("usage")
            )

            response = result.get(
                "response"
            )

            if isinstance(
                response,
                dict,
            ):

                candidates.append(
                    response.get("usage")
                )

        for usage in candidates:

            if not isinstance(
                usage,
                dict,
            ):
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
    # ERROR EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_error(
        data: dict[str, Any],
    ) -> str | None:

        errors = data.get(
            "errors"
        )

        if isinstance(
            errors,
            list,
        ):

            messages: list[str] = []

            for error in errors:

                if isinstance(
                    error,
                    dict,
                ):

                    message = error.get(
                        "message"
                    )

                    if isinstance(
                        message,
                        str,
                    ):
                        messages.append(
                            message
                        )

                elif isinstance(
                    error,
                    str,
                ):

                    messages.append(
                        error
                    )

            if messages:
                return "; ".join(
                    messages
                )

        error = data.get(
            "error"
        )

        if isinstance(
            error,
            str,
        ):
            return error

        if isinstance(
            error,
            dict,
        ):

            message = error.get(
                "message"
            )

            if isinstance(
                message,
                str,
            ):
                return message

        return None

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

        url = self._endpoint(
            model
        )

        payload = self._build_payload(
            request
        )

        print(
            "CLOUDFLARE PAYLOAD DEBUG:",
            {
                "request_id": request.metadata.get(
                    "request_id"
                ),
                "model": model,
                "messages": len(
                    payload.get(
                        "messages",
                        [],
                    )
                ),
                "content_types": [
                    type(
                        message.get("content")
                    ).__name__
                    for message in payload.get(
                        "messages",
                        [],
                    )
                ],
                "tools": len(
                    payload.get(
                        "tools",
                        [],
                    )
                ),
                "tool_choice": payload.get(
                    "tool_choice"
                ),
            },
            flush=True,
        )

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

                usage = (
                    self._extract_usage(
                        data
                    )
                )

            except ValueError:

                data = {}

                usage = ProviderUsage()

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            if response.status_code >= 400:

                error_message = (
                    self._extract_error(
                        data
                    )
                    or response.text
                    or (
                        f"HTTP "
                        f"{response.status_code}"
                    )
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

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = self._extract_content(
                data
            )

            tool_calls = (
                self._extract_tool_calls(
                    data
                )
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
                    error=(
                        "Cloudflare returned "
                        "an empty response."
                    ),
                    elapsed=elapsed,
                    http_status=response.status_code,
                    raw_response=data,
                    usage=usage,
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
                usage=usage,
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
                error=(
                    str(exc)
                    or "Cloudflare request timed out."
                ),
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
                error=(
                    str(exc)
                    or "Unexpected Cloudflare provider failure."
                ),
                elapsed=elapsed,
            )