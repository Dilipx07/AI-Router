"""
AI Router v1
Google Gemini provider adapter.

Implements the BaseProvider interface using the
Google Generative Language REST API.
"""

from __future__ import annotations

import json
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
from router.ids import create_tool_call_id


class GoogleProvider(BaseProvider):
    """
    Google Gemini provider.

    Configuration:
        api_key: Google Gemini API key
        base_url: Google Generative Language API base URL
        timeout: Request timeout in seconds
    """

    name = "google"

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
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config=config)

        self.api_key = str(
            self.config.get("api_key", "")
        ).strip()

        self.base_url = str(
            self.config.get(
                "base_url",
                "https://generativelanguage.googleapis.com/v1beta/models",
            )
        ).rstrip("/")

        self.timeout = float(
            self.config.get("timeout", 60)
        )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(self) -> bool:
        """
        Return True when the Google API key is configured.
        """

        return bool(self.api_key)

    # ========================================================
    # ENDPOINT
    # ========================================================

    def _endpoint(
        self,
        model: str,
    ) -> str:
        """
        Build the Gemini generateContent endpoint.
        """

        return (
            f"{self.base_url}/"
            f"{model}:generateContent"
        )

    # ========================================================
    # HEADERS
    # ========================================================

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ========================================================
    # MESSAGE CONVERSION
    # ========================================================

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Convert OpenAI-style messages into Gemini contents.

        Returns:
            contents, system_instruction
        """

        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for message in messages:

            role = str(
                message.get("role", "user")
            ).lower()

            content = message.get(
                "content",
                "",
            )

            # ------------------------------------------------
            # SYSTEM
            # ------------------------------------------------

            if role == "system":

                if isinstance(content, str):

                    if system_instruction:
                        system_instruction += (
                            "\n" + content
                        )
                    else:
                        system_instruction = content

                elif isinstance(content, list):

                    parts = []

                    for item in content:

                        if not isinstance(item, dict):
                            continue

                        text = item.get("text")

                        if isinstance(text, str):
                            parts.append(text)

                    if parts:

                        text = "\n".join(parts)

                        if system_instruction:
                            system_instruction += (
                                "\n" + text
                            )
                        else:
                            system_instruction = text

                continue

            # ------------------------------------------------
            # ASSISTANT
            # ------------------------------------------------

            if role == "assistant":
                gemini_role = "model"

            else:
                gemini_role = "user"

            parts = GoogleProvider._content_to_parts(
                content
            )

            if not parts:
                parts = [
                    {
                        "text": "",
                    }
                ]

            contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )

        return contents, system_instruction

    # ========================================================
    # CONTENT CONVERSION
    # ========================================================

    @staticmethod
    def _content_to_parts(
        content: Any,
    ) -> list[dict[str, Any]]:
        """
        Convert common message content formats to Gemini parts.
        """

        if isinstance(content, str):

            return [
                {
                    "text": content,
                }
            ]

        if not isinstance(content, list):
            return []

        parts: list[dict[str, Any]] = []

        for item in content:

            if isinstance(item, str):

                parts.append(
                    {
                        "text": item,
                    }
                )

                continue

            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            # ----------------------------------------------
            # TEXT
            # ----------------------------------------------

            if item_type == "text":

                text = item.get("text")

                if isinstance(text, str):

                    parts.append(
                        {
                            "text": text,
                        }
                    )

                continue

            # ----------------------------------------------
            # IMAGE URL
            # ----------------------------------------------

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

                    # Gemini accepts inline image data.
                    # Remote URLs cannot be passed directly
                    # through generateContent.
                    if url.startswith(
                        "data:"
                    ):

                        try:

                            header, data = url.split(
                                ",",
                                1,
                            )

                            mime_type = (
                                header.split(
                                    ";",
                                    1,
                                )[0]
                                .replace(
                                    "data:",
                                    "",
                                )
                            )

                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": data,
                                    }
                                }
                            )

                        except ValueError:
                            pass

                continue

        return parts

    # ========================================================
    # TOOL CONVERSION
    # ========================================================

    @staticmethod
    def _convert_tools(
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """
        Convert OpenAI-style function tools to Gemini tools.
        """

        if not tools:
            return []

        declarations: list[dict[str, Any]] = []

        for tool in tools:

            if not isinstance(tool, dict):
                continue

            function = tool.get("function")

            if not isinstance(function, dict):
                continue

            declaration: dict[str, Any] = {
                "name": function.get(
                    "name",
                    "",
                ),
            }

            description = function.get(
                "description"
            )

            if description:
                declaration["description"] = (
                    description
                )

            parameters = function.get(
                "parameters"
            )

            if isinstance(parameters, dict):
                declaration["parameters"] = (
                    parameters
                )

            declarations.append(
                declaration
            )

        if not declarations:
            return []

        return [
            {
                "function_declarations":
                    declarations,
            }
        ]

    # ========================================================
    # GENERATION CONFIG
    # ========================================================

    @staticmethod
    def _generation_config(
        request: ProviderRequest,
    ) -> dict[str, Any]:
        """
        Build Gemini generation configuration.
        """

        config: dict[str, Any] = {}

        if request.temperature is not None:
            config["temperature"] = (
                request.temperature
            )

        if request.top_p is not None:
            config["topP"] = request.top_p
            
        if request.max_tokens is not None:
            config["maxOutputTokens"] = (
                request.max_tokens
            )

        return config

    # ========================================================
    # REQUEST PAYLOAD
    # ========================================================

    @staticmethod
    def _build_payload(
        request: ProviderRequest,
    ) -> dict[str, Any]:
        """
        Convert ProviderRequest into Gemini format.
        """

        contents, system_instruction = (
            GoogleProvider._convert_messages(
                request.messages
            )
        )

        payload: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:

            payload["systemInstruction"] = {
                "parts": [
                    {
                        "text": system_instruction,
                    }
                ]
            }

        tools = GoogleProvider._convert_tools(
            request.tools
        )

        if tools:
            payload["tools"] = tools

        generation_config = (
            GoogleProvider._generation_config(
                request
            )
        )

        if generation_config:
            payload["generationConfig"] = (
                generation_config
            )

        if request.tool_choice is not None:

            tool_config: dict[str, Any] = {}

            if isinstance(
                request.tool_choice,
                str,
            ):

                if request.tool_choice == "none":

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "NONE",
                    }

                elif request.tool_choice == "required":

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "ANY",
                    }

                elif request.tool_choice == "auto":

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "AUTO",
                    }

            if tool_config:
                payload["toolConfig"] = tool_config

        return payload

    # ========================================================
    # RESPONSE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_content(
        data: dict[str, Any],
    ) -> str:
        """
        Extract textual content from Gemini response.
        """

        candidates = data.get(
            "candidates",
            []
        )

        if not isinstance(
            candidates,
            list,
        ):
            return ""

        texts: list[str] = []

        for candidate in candidates:

            if not isinstance(
                candidate,
                dict,
            ):
                continue

            content = candidate.get(
                "content"
            )

            if not isinstance(
                content,
                dict,
            ):
                continue

            parts = content.get(
                "parts",
                []
            )

            if not isinstance(
                parts,
                list,
            ):
                continue

            for part in parts:

                if not isinstance(
                    part,
                    dict,
                ):
                    continue

                text = part.get(
                    "text"
                )

                if isinstance(
                    text,
                    str,
                ):
                    texts.append(text)

        return "".join(texts)

    # ========================================================
    # TOOL CALL EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_tool_calls(
        data: dict[str, Any],
        request_id : str,   
    ) -> list[dict[str, Any]]:
        """
        Extract Gemini function calls and normalize them.
        """

        candidates = data.get(
            "candidates",
            []
        )

        if not isinstance(
            candidates,
            list,
        ):
            return []

        tool_calls: list[dict[str, Any]] = []

        for candidate in candidates:

            if not isinstance(
                candidate,
                dict,
            ):
                continue

            content = candidate.get(
                "content"
            )

            if not isinstance(
                content,
                dict,
            ):
                continue

            parts = content.get(
                "parts",
                []
            )

            if not isinstance(
                parts,
                list,
            ):
                continue

            for index, part in enumerate(parts):

                if not isinstance(
                    part,
                    dict,
                ):
                    continue

                function_call = part.get(
                    "functionCall"
                )

                if not isinstance(
                    function_call,
                    dict,
                ):
                    continue

                name = function_call.get(
                    "name",
                    "",
                )

                args = function_call.get(
                    "args",
                    {},
                )

                tool_calls.append(
                    {
                        "id": create_tool_call_id(
                            request_id,
                            index,
                        ),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                args
                                if isinstance(args, dict)
                                else {},
                                separators=(",", ":"),
                            ),
                        },
                    }
                )

        return tool_calls

    # ========================================================
    # FINISH REASON
    # ========================================================

    @staticmethod
    def _extract_finish_reason(
        data: dict[str, Any],
    ) -> str | None:
        """
        Extract Gemini finish reason.
        """

        candidates = data.get(
            "candidates",
            []
        )

        if not isinstance(
            candidates,
            list,
        ):
            return None

        for candidate in candidates:

            if not isinstance(
                candidate,
                dict,
            ):
                continue

            reason = candidate.get(
                "finishReason"
            )

            if isinstance(
                reason,
                str,
            ):
                return reason

        return None

    # ========================================================
    # USAGE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_usage(
        data: dict[str, Any],
    ) -> ProviderUsage:
        """
        Extract and normalize Gemini usage metadata.
        """

        usage = data.get(
            "usageMetadata",
            {},
        )

        if not isinstance(
            usage,
            dict,
        ):
            return ProviderUsage()

        return ProviderUsage(
            prompt_tokens=usage.get(
                "promptTokenCount",
                0,
            ) or 0,

            completion_tokens=usage.get(
                "candidatesTokenCount",
                0,
            ) or 0,

            total_tokens=usage.get(
                "totalTokenCount",
                0,
            ) or 0,
        ).normalized()
    # ========================================================
    # ERROR EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_error(
        data: dict[str, Any],
    ) -> str | None:
        """
        Extract Google's API error message.
        """

        error = data.get(
            "error"
        )

        if not isinstance(
            error,
            dict,
        ):
            return None

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
        Execute a Gemini generateContent request.
        """

        if not self.is_available():

            return ProviderResult.failure_result(
                provider=self.provider_name,
                model=model,
                status="auth_failure",
                error=(
                    "Google provider is not configured. "
                    "GOOGLE_API_KEY is required."
                ),
            )

        started = time.perf_counter()

        url = self._endpoint(model)

        params = {
            "key": self.api_key,
        }

        payload = self._build_payload(
            request
        )

        try:

            response = httpx.post(
                url,
                params=params,
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
                usage = self._extract_usage(
                    data
                )
            except ValueError:
                data = {}

            if response.status_code >= 400:

                error_message = (
                    self._extract_error(data)
                    or response.text
                    or f"HTTP {response.status_code}"
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

            content = self._extract_content(
                data
            )

            request_id = str(
                request.metadata.get(
                    "request_id",
                    "",
                )
            ).strip()

            if not request_id:
                raise ValueError(
                    "ProviderRequest is missing request_id."
                )

            tool_calls = (
                self._extract_tool_calls(
                    data,
                    request_id,
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
                        "Google returned an empty response."
                    ),
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
                error=(
                    str(exc)
                    or "Google request timed out."
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
                error=str(exc),
                elapsed=elapsed,
            )