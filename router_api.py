#!/usr/bin/env python3
"""
router_api.py

OpenAI-compatible HTTP API for the AI Router.

Endpoints:
    GET  /health
    GET  /metrics
    GET  /quota
    GET  /v1/models
    POST /v1/chat/completions

Designed for OpenAI-compatible clients and agentic clients
that use tool calling.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from providers.base import ProviderRequest, ProviderResult
from router.errors import ErrorType
from router.executor import ExecutionError
from router.ids import create_request_id
from router.runtime import RUNTIME


# ============================================================
# HTTP ERROR MAPPING
# ============================================================

ERROR_HTTP_STATUS: dict[ErrorType, int] = {
    ErrorType.INVALID_REQUEST: 400,
    ErrorType.TOOL_UNSUPPORTED: 400,
    ErrorType.CONTEXT_TOO_LARGE: 400,
    ErrorType.CONTENT_FILTER: 400,

    ErrorType.AUTH_FAILURE: 401,

    ErrorType.PAYMENT_REQUIRED: 402,

    ErrorType.MODEL_NOT_FOUND: 404,

    ErrorType.RATE_LIMIT: 429,
    ErrorType.QUOTA_EXCEEDED: 429,

    ErrorType.TIMEOUT: 504,

    ErrorType.NETWORK_FAILURE: 502,
    ErrorType.PROVIDER_FAILURE: 502,
    ErrorType.EMPTY_RESPONSE: 502,

    ErrorType.MODEL_UNAVAILABLE: 503,

    ErrorType.UNKNOWN: 500,
}


def error_http_status(
    error: ExecutionError,
) -> int:
    """
    Convert a normalized router error into its HTTP status.
    """

    return ERROR_HTTP_STATUS.get(
        error.error_type,
        500,
    )


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Manage the production runtime lifecycle.
    """

    RUNTIME.startup()

    try:
        yield

    finally:
        RUNTIME.shutdown()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Router",
    version="7.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EXECUTION COMPATIBILITY WRAPPER
# ============================================================

def execute_request(
    request: ProviderRequest,
    *,
    required_capabilities: set[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProviderResult:
    """
    Execute a request through the shared runtime executor.

    Kept as a module-level wrapper so existing API regression
    tests and integrations can intercept the execution boundary
    without bypassing the production runtime.
    """

    return RUNTIME.executor.route_and_execute(
        request,
        required_capabilities=(
            required_capabilities
            or set()
        ),
        provider=provider,
        model=model,
        metadata=metadata,
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow"
    )

    model: str = "ai-router"

    messages: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    tools: Optional[
        list[dict[str, Any]]
    ] = None

    tool_choice: Any = None

    stream: bool = False

    temperature: Optional[float] = None

    max_tokens: Optional[int] = None

    top_p: Optional[float] = None

    provider: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def make_id(
    prefix: str,
) -> str:
    return (
        f"{prefix}-"
        f"{uuid.uuid4().hex}"
    )


# ============================================================
# OPENAI RESPONSE
# ============================================================

def make_chat_response(
    result: ProviderResult,
    requested_model: str,
    request_id: str,
) -> dict:
    """
    Convert normalized ProviderResult into an
    OpenAI-compatible chat.completion response.
    """

    completion_id = make_id(
        "chatcmpl"
    )

    created = int(
        time.time()
    )

    content = (
        result.content
        or ""
    )

    tool_calls = (
        result.tool_calls
        or []
    )

    finish_reason = (
        result.finish_reason
    )

    if tool_calls:

        finish_reason = (
            "tool_calls"
        )

    elif not finish_reason:

        finish_reason = "stop"

    message: dict[
        str,
        Any,
    ] = {
        "role": "assistant",
        "content": content,
    }

    if tool_calls:

        message[
            "tool_calls"
        ] = tool_calls

    return {
        "id": completion_id,

        "object":
            "chat.completion",

        "created": created,

        "model":
            requested_model
            or "ai-router",

        "choices": [
            {
                "index": 0,

                "message":
                    message,

                "finish_reason":
                    finish_reason,
            }
        ],

        "usage": {
            "prompt_tokens":
                result.usage.prompt_tokens,

            "completion_tokens":
                result.usage.completion_tokens,

            "total_tokens":
                result.usage.total_tokens,
        },

        "x_router": {
            "provider":
                result.provider,

            "model":
                result.model,

            "elapsed":
                result.elapsed,

            "request_id":
                request_id,
        },
    }


# ============================================================
# STREAMING
# ============================================================

def sse(
    data: dict,
) -> str:
    """
    Encode one SSE event.
    """

    return (
        "data: "
        + json.dumps(
            data,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n\n"
    )


def make_stream_chunks(
    response: dict,
):
    """
    Convert the completed non-streaming response into
    OpenAI-compatible SSE chunks.
    """

    completion_id = (
        response["id"]
    )

    created = (
        response["created"]
    )

    model = (
        response["model"]
    )

    usage = response.get(
        "usage",
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    choice = (
        response["choices"][0]
    )

    message = (
        choice["message"]
    )

    finish_reason = (
        choice["finish_reason"]
    )

    # --------------------------------------------------------
    # INITIAL ROLE CHUNK
    # --------------------------------------------------------

    yield sse(
        {
            "id": completion_id,
            "object":
                "chat.completion.chunk",
            "created": created,
            "model": model,

            "choices": [
                {
                    "index": 0,

                    "delta": {
                        "role":
                            "assistant",
                    },

                    "finish_reason":
                        None,
                }
            ],
        }
    )

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content = (
        message.get("content")
    )

    if content:

        yield sse(
            {
                "id":
                    completion_id,

                "object":
                    "chat.completion.chunk",

                "created":
                    created,

                "model":
                    model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "content":
                                content,
                        },

                        "finish_reason":
                            None,
                    }
                ],
            }
        )

    # --------------------------------------------------------
    # TOOL CALLS
    # --------------------------------------------------------

    tool_calls = (
        message.get(
            "tool_calls"
        )
        or []
    )

    for index, tool_call in enumerate(
        tool_calls
    ):

        function = (
            tool_call.get(
                "function",
                {},
            )
        )

        tool_index = (
            tool_call.get(
                "index",
                index,
            )
        )

        yield sse(
            {
                "id":
                    completion_id,

                "object":
                    "chat.completion.chunk",

                "created":
                    created,

                "model":
                    model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "tool_calls": [
                                {
                                    "index":
                                        tool_index,

                                    "id":
                                        tool_call.get(
                                            "id"
                                        ),

                                    "type":
                                        "function",

                                    "function": {
                                        "name":
                                            function.get(
                                                "name",
                                                "",
                                            ),
                                    },
                                }
                            ]
                        },

                        "finish_reason":
                            None,
                    }
                ],
            }
        )

        arguments = (
            function.get(
                "arguments",
                "",
            )
        )

        if arguments:

            yield sse(
                {
                    "id":
                        completion_id,

                    "object":
                        "chat.completion.chunk",

                    "created":
                        created,

                    "model":
                        model,

                    "choices": [
                        {
                            "index": 0,

                            "delta": {
                                "tool_calls": [
                                    {
                                        "index":
                                            tool_index,

                                        "function": {
                                            "arguments":
                                                arguments,
                                        },
                                    }
                                ]
                            },

                            "finish_reason":
                                None,
                        }
                    ],
                }
            )

    # --------------------------------------------------------
    # FINAL CHUNK
    # --------------------------------------------------------

    yield sse(
        {
            "id":
                completion_id,

            "object":
                "chat.completion.chunk",

            "created":
                created,

            "model":
                model,

            "choices": [
                {
                    "index": 0,

                    "delta": {},

                    "finish_reason":
                        finish_reason,
                }
            ],

            "usage":
                usage,
        }
    )

    yield "data: [DONE]\n\n"


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
def metrics():
    return {
        "object": "metrics",

        "data": [
            {
                "provider":
                    record.provider,

                "model":
                    record.model,

                "requests":
                    record.requests,

                "successes":
                    record.successes,

                "failures":
                    record.failures,

                "prompt_tokens":
                    record.prompt_tokens,

                "completion_tokens":
                    record.completion_tokens,

                "total_tokens":
                    record.total_tokens,

                "total_latency":
                    record.total_latency,

                "average_latency":
                    record.average_latency,

                "success_rate":
                    record.success_rate,

                "failure_rate":
                    record.failure_rate,

                "last_latency":
                    record.last_latency,

                "last_success_at":
                    record.last_success_at,

                "last_failure_at":
                    record.last_failure_at,

                "last_error_type":
                    record.last_error_type,
            }

            for record
            in RUNTIME.metrics.snapshot()
        ],
    }


# ============================================================
# QUOTA
# ============================================================

@app.get("/quota")
def quota():
    return {
        "object": "quota",

        "data": [
            {
                "provider":
                    state.provider,

                "model":
                    state.model,

                "requests":
                    state.requests,

                "prompt_tokens":
                    state.prompt_tokens,

                "completion_tokens":
                    state.completion_tokens,

                "total_tokens":
                    state.total_tokens,

                "first_recorded_at":
                    state.first_recorded_at,

                "last_recorded_at":
                    state.last_recorded_at,
            }

            for state
            in RUNTIME.quota.snapshot()
        ],
    }


# ============================================================
# MODELS
# ============================================================

@app.get("/v1/models")
def models():

    return {
        "object": "list",

        "data": [
            {
                "id":
                    "ai-router",

                "object":
                    "model",

                "owned_by":
                    "local-router",
            }
        ],
    }


# ============================================================
# CHAT COMPLETIONS
# ============================================================

@app.post(
    "/v1/chat/completions"
)
def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
):
    # --------------------------------------------------------
    # REQUEST ID
    # --------------------------------------------------------

    request_id = (
        http_request.headers.get(
            "X-Request-ID"
        )
        or create_request_id()
    )

    # --------------------------------------------------------
    # VALIDATE MESSAGES
    # --------------------------------------------------------

    if not request.messages:

        return JSONResponse(
            status_code=400,

            content={
                "error": {
                    "message":
                        "messages must not be empty",

                    "type":
                        "invalid_request_error",

                    "code":
                        "invalid_request_error",
                }
            },

            headers={
                "X-Request-ID":
                    request_id,
            },
        )

    # --------------------------------------------------------
    # PROVIDER REQUEST
    # --------------------------------------------------------

    provider_request = ProviderRequest(
        messages=request.messages,

        tools=request.tools,

        tool_choice=request.tool_choice,

        temperature=request.temperature,

        max_tokens=request.max_tokens,

        top_p=request.top_p,

        metadata={
            "api_model":
                request.model,

            "request_id":
                request_id,
        },
    )

    # --------------------------------------------------------
    # REQUIRED CAPABILITIES
    # --------------------------------------------------------

    required_capabilities: set[
        str
    ] = set()

    if request.tools:

        required_capabilities.add(
            "tools"
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    try:

        result = execute_request(
            provider_request,
            required_capabilities=required_capabilities,
            provider=request.provider,
        )

    except ExecutionError as error:

        status_code = (
            error_http_status(
                error
            )
        )

        usage = error.usage

        resolved_request_id = (
            error.metadata.get(
                "request_id"
            )
            or request_id
        )

        return JSONResponse(
            status_code=status_code,

            content={
                "error": {
                    "message":
                        error.message,

                    "type":
                        error.error_type.value,

                    "code":
                        error.error_type.value,
                },

                "x_router": {
                    "provider":
                        error.provider,

                    "model":
                        error.model,

                    "retryable":
                        error.retryable,

                    "request_id":
                        resolved_request_id,

                    "attempts":
                        error.metadata.get(
                            "attempts",
                            [],
                        ),

                    "routing_rejected":
                        error.metadata.get(
                            "routing_rejected",
                            [],
                        ),

                    "usage": {
                        "prompt_tokens":
                            usage.prompt_tokens,

                        "completion_tokens":
                            usage.completion_tokens,

                        "total_tokens":
                            usage.total_tokens,
                    },
                },
            },

            headers={
                "X-Request-ID":
                    resolved_request_id,
            },
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response = make_chat_response(
        result,
        request.model,
        request_id,
    )

    # --------------------------------------------------------
    # STREAMING
    # --------------------------------------------------------

    if request.stream:

        return StreamingResponse(
            make_stream_chunks(
                response
            ),

            media_type=
                "text/event-stream",

            headers={
                "Cache-Control":
                    "no-cache",

                "Connection":
                    "keep-alive",

                "X-Accel-Buffering":
                    "no",

                "X-Request-ID":
                    request_id,
            },
        )

    # --------------------------------------------------------
    # NON-STREAMING
    # --------------------------------------------------------

    return JSONResponse(
        content=response,

        media_type=
            "application/json",

        headers={
            "X-Request-ID":
                request_id,
        },
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "router_api:app",

        host="127.0.0.1",

        port=48765,

        reload=False,
    )
