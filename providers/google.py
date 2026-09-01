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
            self.config.get(
                "api_key",
                "",
            )
        ).strip()

        self.base_url = str(
            self.config.get(
                "base_url",
                (
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/models"
                ),
            )
        ).rstrip("/")

        self.timeout = float(
            self.config.get(
                "timeout",
                60,
            )
        )

    # ========================================================
    # AVAILABILITY
    # ========================================================

    def is_available(self) -> bool:
        return bool(
            self.api_key
        )

    # ========================================================
    # ENDPOINT
    # ========================================================

    def _endpoint(
        self,
        model: str,
    ) -> str:
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
    ) -> tuple[
        list[dict[str, Any]],
        str | None,
    ]:

        contents: list[
            dict[str, Any]
        ] = []

        system_instruction: str | None = None

        for message in messages:

            role = str(
                message.get(
                    "role",
                    "user",
                )
            ).lower()

            content = message.get(
                "content",
                "",
            )

            # ------------------------------------------------
            # SYSTEM
            # ------------------------------------------------

            if role == "system":

                if isinstance(
                    content,
                    str,
                ):

                    if system_instruction:

                        system_instruction += (
                            "\n" + content
                        )

                    else:

                        system_instruction = content

                elif isinstance(
                    content,
                    list,
                ):

                    parts: list[str] = []

                    for item in content:

                        if not isinstance(
                            item,
                            dict,
                        ):
                            continue

                        text = item.get(
                            "text"
                        )

                        if isinstance(
                            text,
                            str,
                        ):
                            parts.append(
                                text
                            )

                    if parts:

                        text = "\n".join(
                            parts
                        )

                        if system_instruction:

                            system_instruction += (
                                "\n" + text
                            )

                        else:

                            system_instruction = text

                continue

            gemini_role = (
                "model"
                if role == "assistant"
                else "user"
            )

            parts = (
                GoogleProvider._content_to_parts(
                    content
                )
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

        return (
            contents,
            system_instruction,
        )

    # ========================================================
    # CONTENT CONVERSION
    # ========================================================

    @staticmethod
    def _content_to_parts(
        content: Any,
    ) -> list[dict[str, Any]]:

        if isinstance(
            content,
            str,
        ):

            return [
                {
                    "text": content,
                }
            ]

        if not isinstance(
            content,
            list,
        ):
            return []

        parts: list[
            dict[str, Any]
        ] = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                parts.append(
                    {
                        "text": item,
                    }
                )

                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            item_type = item.get(
                "type"
            )

            # ------------------------------------------------
            # TEXT
            # ------------------------------------------------

            if item_type == "text":

                text = item.get(
                    "text"
                )

                if isinstance(
                    text,
                    str,
                ):

                    parts.append(
                        {
                            "text": text,
                        }
                    )

                continue

            # ------------------------------------------------
            # IMAGE URL
            # ------------------------------------------------

            if item_type == "image_url":

                image_url = item.get(
                    "image_url"
                )

                if isinstance(
                    image_url,
                    dict,
                ):

                    url = image_url.get(
                        "url"
                    )

                else:

                    url = image_url

                if isinstance(
                    url,
                    str,
                ) and url.startswith(
                    "data:"
                ):

                    try:

                        header, data = (
                            url.split(
                                ",",
                                1,
                            )
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
                                    "mime_type":
                                        mime_type,
                                    "data":
                                        data,
                                }
                            }
                        )

                    except ValueError:
                        pass

                continue

        return parts

    # ========================================================
    # GEMINI SCHEMA NORMALIZATION
    # ========================================================

    @classmethod
    def _sanitize_schema(
        cls,
        schema: Any,
    ) -> Any:
        """
        Convert OpenAI/Cline JSON Schema into the subset
        accepted by Gemini FunctionDeclaration Schema.

        Gemini does not accept arbitrary JSON Schema keywords.
        Required fields must also exist in the corresponding
        properties object.
        """

        if isinstance(schema, list):

            return [
                cls._sanitize_schema(item)
                for item in schema
            ]

        if not isinstance(
            schema,
            dict,
        ):
            return schema

        # Gemini-compatible schema fields.
        allowed_keys = {
            "type",
            "description",
            "properties",
            "required",
            "items",
            "enum",
            "format",
            "minimum",
            "maximum",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "pattern",
            "propertyOrdering",
        }

        result: dict[str, Any] = {}

        # ----------------------------------------------------
        # NORMAL FIELDS
        # ----------------------------------------------------

        for key, value in schema.items():

            if key not in allowed_keys:
                continue

            result[key] = cls._sanitize_schema(
                value
            )

        # ----------------------------------------------------
        # anyOf / oneOf
        # ----------------------------------------------------

        for union_name in (
            "anyOf",
            "oneOf",
        ):

            union = schema.get(
                union_name
            )

            if not isinstance(
                union,
                list,
            ):
                continue

            usable = []

            for item in union:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                if item.get("type") == "null":
                    continue

                usable.append(item)

            if len(usable) == 1:

                normalized = cls._sanitize_schema(
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
        # const -> enum
        # ----------------------------------------------------

        if "const" in schema:

            value = schema.get(
                "const"
            )

            if "enum" not in result:

                result["enum"] = [
                    value
                ]

        # ----------------------------------------------------
        # PROPERTIES
        # ----------------------------------------------------

        original_properties = schema.get(
            "properties"
        )

        if isinstance(
            original_properties,
            dict,
        ):

            cleaned_properties: dict[
                str,
                Any,
            ] = {}

            for name, value in original_properties.items():

                if not isinstance(
                    name,
                    str,
                ):
                    continue

                cleaned = cls._sanitize_schema(
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
        #
        # Gemini requires every required field to exist in
        # properties.
        # ----------------------------------------------------

        original_required = schema.get(
            "required"
        )

        final_properties = result.get(
            "properties"
        )

        if isinstance(
            original_required,
            list,
        ) and isinstance(
            final_properties,
            dict,
        ):

            result["required"] = [
                name
                for name in original_required
                if (
                    isinstance(
                        name,
                        str,
                    )
                    and name in final_properties
                )
            ]

            # Do not send an empty required list.
            if not result["required"]:

                result.pop(
                    "required",
                    None,
                )

        else:

            result.pop(
                "required",
                None,
            )

        # ----------------------------------------------------
        # ITEMS
        # ----------------------------------------------------

        if "items" in result:

            items = result.get(
                "items"
            )

            if isinstance(
                items,
                dict,
            ):

                result["items"] = (
                    cls._sanitize_schema(
                        items
                    )
                )

        return result

    # ========================================================
    # TOOL CONVERSION
    # ========================================================

    @classmethod
    def _convert_tools(
        cls,
        tools: list[
            dict[str, Any]
        ] | None,
    ) -> list[
        dict[str, Any]
    ]:
        """
        Convert OpenAI function tools into Gemini
        function declarations.
        """

        if not tools:
            return []

        declarations: list[
            dict[str, Any]
        ] = []

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
                "name",
                "",
            )

            if not isinstance(
                name,
                str,
            ) or not name:
                continue

            declaration: dict[
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
                declaration[
                    "description"
                ] = description

            parameters = function.get(
                "parameters"
            )

            if isinstance(
                parameters,
                dict,
            ):

                declaration[
                    "parameters"
                ] = cls._sanitize_schema(
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

        config: dict[
            str,
            Any,
        ] = {}

        if request.temperature is not None:

            config[
                "temperature"
            ] = request.temperature

        if request.top_p is not None:

            config[
                "topP"
            ] = request.top_p

        if request.max_tokens is not None:

            config[
                "maxOutputTokens"
            ] = request.max_tokens

        return config

    # ========================================================
    # REQUEST PAYLOAD
    # ========================================================

    @classmethod
    def _build_payload(
        cls,
        request: ProviderRequest,
    ) -> dict[str, Any]:

        (
            contents,
            system_instruction,
        ) = cls._convert_messages(
            request.messages
        )

        payload: dict[
            str,
            Any,
        ] = {
            "contents": contents,
        }

        if system_instruction:

            payload[
                "systemInstruction"
            ] = {
                "parts": [
                    {
                        "text":
                            system_instruction,
                    }
                ]
            }

        tools = cls._convert_tools(
            request.tools
        )

        if tools:

            payload[
                "tools"
            ] = tools

        if request.tool_choice is not None:

            tool_config: dict[
                str,
                Any,
            ] = {}

            if isinstance(
                request.tool_choice,
                str,
            ):

                if (
                    request.tool_choice
                    == "none"
                ):

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "NONE",
                    }

                elif (
                    request.tool_choice
                    == "required"
                ):

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "ANY",
                    }

                elif (
                    request.tool_choice
                    == "auto"
                ):

                    tool_config[
                        "function_calling_config"
                    ] = {
                        "mode": "AUTO",
                    }

            if tool_config:

                payload[
                    "toolConfig"
                ] = tool_config

        generation_config = (
            cls._generation_config(
                request
            )
        )

        if generation_config:

            payload[
                "generationConfig"
            ] = generation_config

        return payload

    # ========================================================
    # RESPONSE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_content(
        data: dict[str, Any],
    ) -> str:

        candidates = data.get(
            "candidates",
            [],
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
                [],
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
                    texts.append(
                        text
                    )

        return "".join(
            texts
        )

    # ========================================================
    # TOOL CALL EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_tool_calls(
        data: dict[str, Any],
        request_id: str,
    ) -> list[dict[str, Any]]:

        candidates = data.get(
            "candidates",
            [],
        )

        if not isinstance(
            candidates,
            list,
        ):
            return []

        tool_calls: list[
            dict[str, Any]
        ] = []

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
                [],
            )

            if not isinstance(
                parts,
                list,
            ):
                continue

            for index, part in enumerate(
                parts
            ):

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
                        "id":
                            create_tool_call_id(
                                request_id,
                                index,
                            ),
                        "type":
                            "function",
                        "function": {
                            "name":
                                name,
                            "arguments":
                                json.dumps(
                                    (
                                        args
                                        if isinstance(
                                            args,
                                            dict,
                                        )
                                        else {}
                                    ),
                                    separators=(
                                        ",",
                                        ":",
                                    ),
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

        candidates = data.get(
            "candidates",
            [],
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
    # USAGE
    # ========================================================

    @staticmethod
    def _extract_usage(
        data: dict[str, Any],
    ) -> ProviderUsage:

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
            prompt_tokens=(
                usage.get(
                    "promptTokenCount",
                    0,
                )
                or 0
            ),
            completion_tokens=(
                usage.get(
                    "candidatesTokenCount",
                    0,
                )
                or 0
            ),
            total_tokens=(
                usage.get(
                    "totalTokenCount",
                    0,
                )
                or 0
            ),
        ).normalized()

    # ========================================================
    # ERROR EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_error(
        data: dict[str, Any],
    ) -> str | None:

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

        url = self._endpoint(
            model
        )

        params = {
            "key": self.api_key,
        }

        payload = self._build_payload(
            request
        )

        print(
            "GOOGLE PAYLOAD DEBUG:",
            {
                "request_id":
                    request.metadata.get(
                        "request_id"
                    ),
                "model":
                    model,
                "messages":
                    len(
                        payload.get(
                            "contents",
                            [],
                        )
                    ),
                "tools":
                    sum(
                        len(
                            tool.get(
                                "function_declarations",
                                [],
                            )
                        )
                        for tool in payload.get(
                            "tools",
                            [],
                        )
                        if isinstance(
                            tool,
                            dict,
                        )
                    ),
                "tool_choice":
                    request.tool_choice,
            },
            flush=True,
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
                    status="provider_failure",
                    error=error_message,
                    elapsed=elapsed,
                    http_status=(
                        response.status_code
                    ),
                    raw_response=data,
                    usage=usage,
                )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = (
                self._extract_content(
                    data
                )
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
                        "Google returned "
                        "an empty response."
                    ),
                    elapsed=elapsed,
                    http_status=(
                        response.status_code
                    ),
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
                http_status=(
                    response.status_code
                ),
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
                error=(
                    str(exc)
                    or "Unexpected Google provider failure."
                ),
                elapsed=elapsed,
            )