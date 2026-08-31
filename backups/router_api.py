#!/usr/bin/env python3
"""
router_api.py

OpenAI-compatible HTTP API for token_router.py.

Endpoints:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions

Designed for Continue, OpenAI-compatible clients,
and agentic clients that use tool calling.
"""

import json
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from token_router import route


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Router",
    version="7.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    Extra fields are accepted so clients such as Continue
    can send provider/client-specific parameters without
    breaking validation.
    """

    model_config = ConfigDict(extra="allow")

    model: str = "ai-router"

    messages: list[dict[str, Any]] = Field(
        default_factory=list
    )

    tools: Optional[list[dict[str, Any]]] = None

    tool_choice: Any = None

    stream: bool = False

    temperature: Optional[float] = None

    max_tokens: Optional[int] = None

    top_p: Optional[float] = None


# ============================================================
# HELPERS
# ============================================================

def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


# ============================================================
# OPENAI RESPONSE
# ============================================================

def make_chat_response(
    result: dict,
    requested_model: str,
) -> dict:
    """
    Convert internal router result into an
    OpenAI-compatible chat.completion response.
    """

    completion_id = make_id("chatcmpl")

    created = int(time.time())

    content = result.get("content") or ""

    tool_calls = result.get("tool_calls") or []

    finish_reason = result.get("finish_reason")

    if tool_calls:
        finish_reason = "tool_calls"

    elif not finish_reason:
        finish_reason = "stop"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": completion_id,

        "object": "chat.completion",

        "created": created,

        # ----------------------------------------------------
        # IMPORTANT:
        # Expose the router model, not the underlying provider.
        # ----------------------------------------------------

        "model": requested_model or "ai-router",

        "choices": [
            {
                "index": 0,

                "message": message,

                "finish_reason": finish_reason,
            }
        ],

        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },

        # ----------------------------------------------------
        # Internal diagnostic metadata.
        # ----------------------------------------------------

        "x_router": {
            "provider": result.get("provider"),

            "model": result.get("model"),

            "elapsed": result.get("elapsed"),
        },
    }


# ============================================================
# STREAMING
# ============================================================

def sse(data: dict) -> str:
    """
    Encode one SSE event as UTF-8 JSON.
    """

    return (
        "data: "
        + json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n\n"
    )


def make_stream_chunks(
    response: dict,
):
    """
    Convert the completed non-streaming response into
    OpenAI-compatible SSE chunks.

    The underlying router is still non-streaming.
    This provides a compatibility layer for clients
    that require stream=true.
    """

    completion_id = response["id"]

    created = response["created"]

    model = response["model"]

    choice = response["choices"][0]

    message = choice["message"]

    finish_reason = choice["finish_reason"]

    # ========================================================
    # INITIAL ROLE CHUNK
    # ========================================================

    yield sse(
        {
            "id": completion_id,

            "object": "chat.completion.chunk",

            "created": created,

            "model": model,

            "choices": [
                {
                    "index": 0,

                    "delta": {
                        "role": "assistant",
                    },

                    "finish_reason": None,
                }
            ],
        }
    )

    # ========================================================
    # CONTENT
    # ========================================================

    content = message.get("content")

    if content:

        yield sse(
            {
                "id": completion_id,

                "object": "chat.completion.chunk",

                "created": created,

                "model": model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "content": content,
                        },

                        "finish_reason": None,
                    }
                ],
            }
        )

    # ========================================================
    # TOOL CALLS
    # ========================================================

    tool_calls = message.get("tool_calls") or []

    for index, tool_call in enumerate(tool_calls):

        function = tool_call.get(
            "function",
            {},
        )

        tool_index = tool_call.get(
            "index",
            index,
        )

        # ----------------------------------------------------
        # Tool-call metadata
        # ----------------------------------------------------

        yield sse(
            {
                "id": completion_id,

                "object": "chat.completion.chunk",

                "created": created,

                "model": model,

                "choices": [
                    {
                        "index": 0,

                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_index,

                                    "id": tool_call.get("id"),

                                    "type": "function",

                                    "function": {
                                        "name": function.get(
                                            "name",
                                            "",
                                        ),
                                    },
                                }
                            ]
                        },

                        "finish_reason": None,
                    }
                ],
            }
        )

        # ----------------------------------------------------
        # Tool arguments
        # ----------------------------------------------------

        arguments = function.get(
            "arguments",
            "",
        )

        if arguments:

            yield sse(
                {
                    "id": completion_id,

                    "object": "chat.completion.chunk",

                    "created": created,

                    "model": model,

                    "choices": [
                        {
                            "index": 0,

                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_index,

                                        "function": {
                                            "arguments": arguments,
                                        },
                                    }
                                ]
                            },

                            "finish_reason": None,
                        }
                    ],
                }
            )

    # ========================================================
    # FINAL CHUNK
    # ========================================================

    yield sse(
        {
            "id": completion_id,

            "object": "chat.completion.chunk",

            "created": created,

            "model": model,

            "choices": [
                {
                    "index": 0,

                    "delta": {},

                    "finish_reason": finish_reason,
                }
            ],
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
# MODELS
# ============================================================

@app.get("/v1/models")
def models():

    return {
        "object": "list",

        "data": [
            {
                "id": "ai-router",

                "object": "model",

                "owned_by": "local-router",
            }
        ],
    }


# ============================================================
# CHAT COMPLETIONS
# ============================================================

@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
):

    # --------------------------------------------------------
    # Validate messages
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
                }
            },
        )

    # --------------------------------------------------------
    # Route request
    # --------------------------------------------------------

    result = route(
        messages=request.messages,

        tools=request.tools,

        tool_choice=request.tool_choice,
    )

    # --------------------------------------------------------
    # Router failure
    # --------------------------------------------------------

    if not result.get("provider"):

        return JSONResponse(
            status_code=503,

            content={
                "error": {
                    "message":
                        result.get(
                            "content",
                            "All providers failed.",
                        ),

                    "type":
                        "router_error",
                },

                "x_router": {
                    "elapsed":
                        result.get("elapsed"),
                },
            },
        )

    # --------------------------------------------------------
    # Build OpenAI response
    # --------------------------------------------------------

    response = make_chat_response(
        result,
        request.model,
    )

    # --------------------------------------------------------
    # Streaming compatibility
    # --------------------------------------------------------

    if request.stream:

        return StreamingResponse(
            make_stream_chunks(response),

            media_type="text/event-stream",

            headers={
                "Cache-Control":
                    "no-cache",

                "Connection":
                    "keep-alive",

                "X-Accel-Buffering":
                    "no",
            },
        )

    return JSONResponse(
        content=response,
        media_type="application/json",
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