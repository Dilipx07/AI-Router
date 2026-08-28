#!/usr/bin/env python3
"""
token_router.py v6

Multi-provider AI router.

Features:
- .env support
- Provider fallbacks
- Model fallbacks
- Model deduplication
- Rate-limit handling
- Authentication handling
- Model-not-found handling
- Payment/quota handling
- Detailed diagnostics
- Request timing
- Correct reporting of the model actually used
- Provider/model override for testing
- Primary-model testing
- Provider model discovery
- Cloudflare OpenAI-compatible API
- Local Ollama support
"""

import argparse
import json
import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


def env(name: str, default: str = "") -> str:
    """Read an environment variable and strip whitespace."""
    return os.getenv(name, default).strip()


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DEFAULT_TIMEOUT = 30
HUGGINGFACE_TIMEOUT = 90
OLLAMA_TIMEOUT = 120

DEFAULT_MAX_TOKENS = 1024

TEST_PROMPT = "Reply with exactly TEST_OK"
TEST_EXPECTED = "TEST_OK"


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # ========================================================
    # HUGGING FACE
    # ========================================================

    "huggingface": {
        "api_key": env("HF_API_KEY"),

        "base_url":
            "https://router.huggingface.co/v1",

        "model":
            "Qwen/Qwen3-Coder-Next",

        "fallback_models": [

            "Qwen/Qwen3-Coder-480B-A35B-Instruct",

            "deepseek-ai/DeepSeek-V4-Flash",

            "moonshotai/Kimi-K2.7-Code",

            "zai-org/GLM-5.2",

            "MiniMaxAI/MiniMax-M3",

            "Qwen/Qwen2.5-Coder-32B-Instruct",

            "Qwen/Qwen2.5-Coder-7B-Instruct",
        ],
    },


    # ========================================================
    # GOOGLE GEMINI
    # ========================================================

    "google": {
        "api_key":
            env("GOOGLE_API_KEY"),

        "base_url":
            "https://generativelanguage.googleapis.com/v1beta/models",

        "model":
            "gemini-3.1-flash-lite",

        "fallback_models": [

            "gemini-3.6-flash",

            "gemini-3.5-flash-lite",

            "gemini-3.5-flash",

            "gemini-2.5-flash",

            "gemini-2.5-flash-lite",
        ],
    },


    # ========================================================
    # GROQ
    # ========================================================

    "groq": {
        "api_key":
            env("GROQ_API_KEY"),

        "base_url":
            "https://api.groq.com/openai/v1",

        # llama-3.3-70b-versatile was shut down
        # by Groq on 2026-08-16.
        "model":
            "openai/gpt-oss-120b",

        "fallback_models": [

            "openai/gpt-oss-20b",

            "qwen/qwen3.6-27b",
        ],
    },


    # ========================================================
    # OPENROUTER
    # ========================================================

    "openrouter": {
        "api_key":
            env("OPENROUTER_API_KEY"),

        "base_url":
            "https://openrouter.ai/api/v1",

        "model":
            "openai/gpt-oss-20b:free",

        "fallback_models": [

            "nvidia/nemotron-3.5-lightning:free",

            "cohere/north-mini-code:free",

            "z-ai/glm-5.2:free",
        ],
    },


    # ========================================================
    # CLOUDFLARE WORKERS AI
    # ========================================================

    "cloudflare": {
        "api_key":
            env("CLOUDFLARE_API_KEY"),

        "account_id":
            env("CLOUDFLARE_ACCOUNT_ID"),

        # OpenAI-compatible Workers AI endpoint.
        "base_url":
            "https://api.cloudflare.com/client/v4/accounts",

        "model":
            "@cf/qwen/qwen3.8-27b",

        "fallback_models": [

            "@cf/zai-org/glm-5.2",

            "@cf/moonshotai/kimi-k2.7-code",

            "@cf/deepseek-ai/deepseek-v4-flash-0731",

            "@cf/deepseek-ai/deepseek-v4-pro-0813",

            "@cf/openai/gpt-oss-120b",

            "@cf/openai/gpt-oss-20b",

            "@cf/qwen/qwen2.5-coder-32b-instruct",

            "@cf/qwen/qwen3-30b-a3b-fp8",

            "@cf/zai-org/glm-4.7-flash",

            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",

            "@cf/meta/llama-3.1-8b-instruct-fp8",
        ],
    },
    
    # ========================================================
    # LOCAL OLLAMA
    # ========================================================

    "ollama": {
        "base_url":
            env(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),

        "model":
            "minimax-m3:cloud",

        "fallback_models": [

            "gpt-oss:120b:cloud",

            "nemotron-3-ultra:cloud",

            "gpt-oss:20b:cloud",

            "nemotron-3-super:cloud",

            "nemotron-3-nano:30b:cloud",

            "gemma4:31b:cloud",
        ],
    },
}


# ============================================================
# PROVIDER ORDER
# ============================================================

PROVIDER_ORDER = [
    "google",
    "groq",
    "openrouter",
    "cloudflare",
    "huggingface",
    "ollama",
]


# ============================================================
# HELPERS
# ============================================================

def provider_has_credentials(provider: str) -> bool:

    cfg = CONFIG[provider]

    if provider == "ollama":
        return True

    if provider == "cloudflare":

        return bool(
            cfg.get("api_key")
            and cfg.get("account_id")
        )

    return bool(
        cfg.get("api_key")
    )


def print_provider_skip(provider: str) -> None:

    cfg = CONFIG[provider]

    if provider == "cloudflare":

        if not cfg.get("api_key"):

            print(
                "  [SKIP] cloudflare: "
                "CLOUDFLARE_API_KEY missing"
            )

        elif not cfg.get("account_id"):

            print(
                "  [SKIP] cloudflare: "
                "CLOUDFLARE_ACCOUNT_ID missing"
            )

        return

    if provider == "ollama":
        return

    print(
        f"  [SKIP] {provider}: API key missing"
    )


def unique_models(
    primary: str,
    fallbacks: list,
) -> list:

    result = []
    seen = set()

    for model in [
        primary,
        *fallbacks,
    ]:

        if not model:
            continue

        if model in seen:
            continue

        seen.add(model)
        result.append(model)

    return result


def build_messages(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Preserve an OpenAI-compatible structured message history.

    Backwards compatible with the original prompt/system_prompt API.
    When messages are supplied, they are passed through without flattening
    or losing tool/tool_calls fields.
    """
    if messages is not None:
        return [dict(message) for message in messages]

    result = []

    if system_prompt:
        result.append({
            "role": "system",
            "content": system_prompt,
        })

    if prompt is not None:
        result.append({
            "role": "user",
            "content": prompt,
        })

    return result



def print_request_time(
    elapsed: float,
) -> None:

    print(
        f"  [TIME] {elapsed:.2f}s"
    )


def extract_openai_message(data: dict) -> dict:
    """Return the first OpenAI-compatible assistant message."""
    choices = data.get("choices", [])
    if not choices:
        return {}

    message = choices[0].get("message", {})
    return message if isinstance(message, dict) else {}


def extract_openai_content(data: dict) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    message = extract_openai_message(data)
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if text:
                parts.append(str(text))
        return "".join(parts).strip()

    return str(content or "").strip()


def extract_openai_tool_calls(data: dict) -> list[dict]:
    """Extract OpenAI-compatible tool_calls without modifying them."""
    message = extract_openai_message(data)
    tool_calls = message.get("tool_calls", [])
    return tool_calls if isinstance(tool_calls, list) else []


def extract_openai_finish_reason(data: dict) -> Optional[str]:
    choices = data.get("choices", [])
    if not choices:
        return None
    return choices[0].get("finish_reason")


def build_openai_payload(
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }

    if tools:
        payload["tools"] = tools

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    return payload



# ============================================================
# GOOGLE GEMINI
# ============================================================

def gemini_tools_from_openai(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Convert OpenAI function tools to Gemini functionDeclarations."""
    if not tools:
        return None

    declarations = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            continue

        function = tool.get("function", {})
        if not isinstance(function, dict):
            continue

        declaration = {
            "name": function.get("name", ""),
        }

        if function.get("description"):
            declaration["description"] = function["description"]

        parameters = function.get("parameters")
        if parameters:
            declaration["parametersJsonSchema"] = parameters

        declarations.append(declaration)

    if not declarations:
        return None

    return [{"functionDeclarations": declarations}]


def gemini_tool_config(tool_choice: Any) -> Optional[dict]:
    """Convert OpenAI tool_choice into Gemini functionCallingConfig."""
    if tool_choice is None:
        return None

    mode = "AUTO"
    allowed = None

    if isinstance(tool_choice, str):
        mapping = {
            "auto": "AUTO",
            "required": "ANY",
            "none": "NONE",
        }
        mode = mapping.get(tool_choice.lower(), "AUTO")

    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function", {})
            name = function.get("name")
            mode = "ANY"
            if name:
                allowed = [name]
        elif tool_choice.get("type") == "auto":
            mode = "AUTO"
        elif tool_choice.get("type") == "required":
            mode = "ANY"
        elif tool_choice.get("type") == "none":
            mode = "NONE"

    config = {"mode": mode}
    if allowed:
        config["allowedFunctionNames"] = allowed

    return {"functionCallingConfig": config}


def call_google(
    model: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> tuple[Optional[str], str, list[dict], Optional[str]]:
    cfg = CONFIG["google"]

    url = f"{cfg['base_url']}/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": cfg["api_key"],
    }

    structured_messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
    )

    contents = []
    system_parts = []

    for message in structured_messages:
        role = message.get("role", "user")
        content = message.get("content")

        # System messages are handled by Gemini's systemInstruction.
        if role == "system":
            if content:
                system_parts.append({"text": str(content)})
            continue

        if role == "tool":
            # Gemini expects tool responses in a functionResponse part.
            tool_name = message.get("name") or "tool"
            tool_content = content
            if isinstance(tool_content, str):
                try:
                    tool_content = json.loads(tool_content)
                except Exception:
                    pass

            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": tool_content if isinstance(tool_content, dict)
                        else {"result": tool_content},
                    }
                }],
            })
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts = []

        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if part.get("text"):
                        parts.append({"text": part["text"]})

        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            if function.get("name"):
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}

                parts.append({
                    "functionCall": {
                        "name": function["name"],
                        "args": arguments if isinstance(arguments, dict) else {},
                    }
                })

        if parts:
            contents.append({
                "role": gemini_role,
                "parts": parts,
            })

    payload = {"contents": contents or [{"role": "user", "parts": [{"text": prompt or ""}]}]}

    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    gemini_tools = gemini_tools_from_openai(tools)
    if gemini_tools:
        payload["tools"] = gemini_tools

    config = gemini_tool_config(tool_choice)
    if config:
        payload["toolConfig"] = config

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        elapsed = time.perf_counter() - start

        print(f"  [DEBUG] google/{model} HTTP {response.status_code}")
        print_request_time(elapsed)

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])

            if not candidates:
                print("  [WARN] Google returned 200 but no candidates")
                print(f"  [DEBUG] {response.text[:1500]}")
                return None, "empty", [], None

            candidate = candidates[0]
            content_obj = candidate.get("content", {})
            parts = content_obj.get("parts", [])

            text_parts = []
            tool_calls = []

            for index, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue

                if part.get("text"):
                    text_parts.append(str(part["text"]))

                function_call = part.get("functionCall")
                if isinstance(function_call, dict) and function_call.get("name"):
                    tool_calls.append({
                        "id": f"call_gemini_{int(time.time() * 1000)}_{index}",
                        "type": "function",
                        "function": {
                            "name": function_call["name"],
                            "arguments": json.dumps(
                                function_call.get("args", {}),
                                separators=(",", ":"),
                            ),
                        },
                    })

            text = "".join(text_parts).strip()
            finish_reason = (
                "tool_calls"
                if tool_calls
                else "stop"
            )

            if not text and not tool_calls:
                return None, "empty", [], finish_reason

            print(f"  [OK] google/{model}")
            return text, "success", tool_calls, finish_reason

        if response.status_code == 429:
            print(f"  [RATE-LIMIT] google/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "rate_limit", [], None

        if response.status_code in (401, 403):
            print(f"  [AUTH-ERROR] google/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "auth_error", [], None

        if response.status_code == 404:
            print(f"  [MODEL-NOT-FOUND] google/{model}")
            return None, "model_not_found", [], None

        print(f"  [FAIL] google/{model}: HTTP {response.status_code}")
        print(f"  [DEBUG] {response.text[:1500]}")
        return None, "error", [], None

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(
            f"  [NETWORK-ERROR] google/{model}: "
            f"{type(exc).__name__}: {exc}"
        )
        print_request_time(elapsed)
        return None, "network_error", [], None



# ============================================================
# OLLAMA NATIVE API
# ============================================================

def call_ollama(
    model: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> tuple[Optional[str], str, list[dict], Optional[str]]:
    """Call Ollama native /api/chat with structured messages and tools."""
    cfg = CONFIG["ollama"]

    url = f"{cfg['base_url']}/api/chat"

    headers = {"Content-Type": "application/json"}

    structured_messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
    )

    payload = {
        "model": model,
        "messages": structured_messages,
        "stream": False,
        "options": {
            "num_predict": DEFAULT_MAX_TOKENS,
        },
    }

    if tools:
        payload["tools"] = tools

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        elapsed = time.perf_counter() - start

        print(f"  [DEBUG] ollama/{model} HTTP {response.status_code}")
        print_request_time(elapsed)

        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            text = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", []) or []

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text and not tool_calls:
                print(f"  [WARN] ollama/{model}: empty response")
                print(f"  [DEBUG] {response.text[:1500]}")
                return None, "empty", [], None

            finish_reason = "tool_calls" if tool_calls else "stop"
            print(f"  [OK] ollama/{model}")
            return text, "success", tool_calls, finish_reason

        if response.status_code == 404:
            print(f"  [MODEL-NOT-FOUND] ollama/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            print("  [OLLAMA] Model is probably not installed or unavailable.")
            print("  [OLLAMA] Run: ollama list")
            print(f"  [OLLAMA] Expected model: {model}")
            return None, "model_not_found", [], None

        if response.status_code in (401, 403):
            print(f"  [AUTH-ERROR] ollama/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "auth_error", [], None

        if response.status_code == 429:
            print(f"  [RATE-LIMIT] ollama/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "rate_limit", [], None

        if response.status_code == 402:
            print(f"  [PAYMENT/QUOTA] ollama/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "payment_required", [], None

        print(f"  [FAIL] ollama/{model}: HTTP {response.status_code}")
        print(f"  [DEBUG] {response.text[:1500]}")
        return None, "error", [], None

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(
            f"  [NETWORK-ERROR] ollama/{model}: "
            f"{type(exc).__name__}: {exc}"
        )
        print_request_time(elapsed)
        return None, "network_error", [], None



# ============================================================
# OPENAI-COMPATIBLE PROVIDERS
# ============================================================

def call_openai_compatible(
    provider: str,
    model: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> tuple[Optional[str], str, list[dict], Optional[str]]:
    cfg = CONFIG[provider]

    url = f"{cfg['base_url']}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }

    structured_messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
    )

    payload = build_openai_payload(
        model=model,
        messages=structured_messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        elapsed = time.perf_counter() - start

        print(f"  [DEBUG] {provider}/{model} HTTP {response.status_code}")
        print_request_time(elapsed)

        if response.status_code == 200:
            data = response.json()
            text = extract_openai_content(data)
            tool_calls = extract_openai_tool_calls(data)
            finish_reason = extract_openai_finish_reason(data)

            if not text and not tool_calls:
                print(f"  [WARN] {provider}/{model}: empty response")
                print(f"  [DEBUG] {response.text[:1500]}")
                return None, "empty", [], finish_reason

            print(f"  [OK] {provider}/{model}")
            return text, "success", tool_calls, finish_reason

        if response.status_code == 429:
            print(f"  [RATE-LIMIT] {provider}/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "rate_limit", [], None

        if response.status_code in (401, 403):
            print(f"  [AUTH-ERROR] {provider}/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "auth_error", [], None

        if response.status_code == 404:
            print(f"  [MODEL-NOT-FOUND] {provider}/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "model_not_found", [], None

        if response.status_code == 402:
            print(f"  [PAYMENT/QUOTA] {provider}/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "payment_required", [], None

        print(f"  [FAIL] {provider}/{model}: HTTP {response.status_code}")
        print(f"  [DEBUG] {response.text[:1500]}")
        return None, "error", [], None

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(
            f"  [NETWORK-ERROR] {provider}/{model}: "
            f"{type(exc).__name__}: {exc}"
        )
        print_request_time(elapsed)
        return None, "network_error", [], None



# ============================================================
# HUGGING FACE
# ============================================================

def call_huggingface(
    model: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> tuple[Optional[str], str, list[dict], Optional[str]]:
    cfg = CONFIG["huggingface"]

    url = f"{cfg['base_url']}/chat/completions"

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    structured_messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
    )

    payload = build_openai_payload(
        model=model,
        messages=structured_messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=HUGGINGFACE_TIMEOUT,
        )

        elapsed = time.perf_counter() - start

        print(f"  [DEBUG] huggingface/{model} HTTP {response.status_code}")
        print_request_time(elapsed)

        if response.status_code == 200:
            data = response.json()
            text = extract_openai_content(data)
            tool_calls = extract_openai_tool_calls(data)
            finish_reason = extract_openai_finish_reason(data)

            if not text and not tool_calls:
                print(f"  [WARN] huggingface/{model}: empty response")
                print(f"  [DEBUG] {response.text[:1500]}")
                return None, "empty", [], finish_reason

            print(f"  [OK] huggingface/{model}")
            return text, "success", tool_calls, finish_reason

        if response.status_code == 429:
            print(f"  [RATE-LIMIT] huggingface/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "rate_limit", [], None

        if response.status_code in (401, 403):
            print(f"  [AUTH-ERROR] huggingface/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "auth_error", [], None

        if response.status_code == 404:
            print(f"  [MODEL-NOT-FOUND] huggingface/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "model_not_found", [], None

        if response.status_code == 402:
            print(f"  [PAYMENT/QUOTA] huggingface/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "payment_required", [], None

        print(f"  [FAIL] huggingface/{model}: HTTP {response.status_code}")
        print(f"  [DEBUG] {response.text[:1500]}")
        return None, "error", [], None

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(
            f"  [NETWORK-ERROR] huggingface/{model}: "
            f"{type(exc).__name__}: {exc}"
        )
        print_request_time(elapsed)
        return None, "network_error", [], None



# ============================================================
# CLOUDFLARE
# ============================================================

def call_cloudflare(
    model: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
) -> tuple[Optional[str], str, list[dict], Optional[str]]:
    cfg = CONFIG["cloudflare"]

    if not cfg.get("api_key"):
        return None, "missing_key", [], None

    if not cfg.get("account_id"):
        return None, "missing_account_id", [], None

    url = (
        f"{cfg['base_url']}/{cfg['account_id']}"
        f"/ai/v1/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }

    structured_messages = build_messages(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
    )

    payload = build_openai_payload(
        model=model,
        messages=structured_messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    start = time.perf_counter()

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        elapsed = time.perf_counter() - start

        print(f"  [DEBUG] cloudflare/{model} HTTP {response.status_code}")
        print_request_time(elapsed)

        if response.status_code == 200:
            data = response.json()
            text = extract_openai_content(data)
            tool_calls = extract_openai_tool_calls(data)
            finish_reason = extract_openai_finish_reason(data)

            if not text and not tool_calls:
                print(f"  [WARN] cloudflare/{model}: empty response")
                print(f"  [DEBUG] {response.text[:2000]}")
                return None, "empty", [], finish_reason

            print(f"  [OK] cloudflare/{model}")
            return text, "success", tool_calls, finish_reason

        if response.status_code == 429:
            print(f"  [RATE-LIMIT] cloudflare/{model}")
            print(f"  [DEBUG] {response.text[:1000]}")
            return None, "rate_limit", [], None

        if response.status_code in (401, 403):
            print(f"  [AUTH-ERROR] cloudflare/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "auth_error", [], None

        if response.status_code == 404:
            print(f"  [MODEL-NOT-FOUND] cloudflare/{model}")
            print(f"  [DEBUG] {response.text[:1500]}")
            return None, "model_not_found", [], None

        print(f"  [FAIL] cloudflare/{model}: HTTP {response.status_code}")
        print(f"  [DEBUG] {response.text[:1500]}")
        return None, "error", [], None

    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(
            f"  [NETWORK-ERROR] cloudflare/{model}: "
            f"{type(exc).__name__}: {exc}"
        )
        print_request_time(elapsed)
        return None, "network_error", [], None



# ============================================================
# PROVIDER CALL
# ============================================================

def call_provider(
    provider: str,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    model_override: Optional[str] = None,
    primary_only: bool = False,
) -> Optional[dict]:
    cfg = CONFIG[provider]

    if not provider_has_credentials(provider):
        print_provider_skip(provider)
        return None

    if model_override:
        models = [model_override]
    elif primary_only:
        models = [cfg["model"]]
    else:
        models = unique_models(
            cfg["model"],
            cfg.get("fallback_models", []),
        )

    print(f"  [MODELS] {len(models)} configured")

    for index, model in enumerate(models, start=1):
        print(f"\n  [MODEL {index}/{len(models)}] {provider}/{model}")

        if provider == "google":
            text, status, tool_calls, finish_reason = call_google(
                model, prompt, system_prompt, messages, tools, tool_choice
            )
        elif provider == "huggingface":
            text, status, tool_calls, finish_reason = call_huggingface(
                model, prompt, system_prompt, messages, tools, tool_choice
            )
        elif provider == "cloudflare":
            text, status, tool_calls, finish_reason = call_cloudflare(
                model, prompt, system_prompt, messages, tools, tool_choice
            )
        elif provider == "ollama":
            text, status, tool_calls, finish_reason = call_ollama(
                model, prompt, system_prompt, messages, tools, tool_choice
            )
        else:
            text, status, tool_calls, finish_reason = call_openai_compatible(
                provider, model, prompt, system_prompt, messages, tools, tool_choice
            )

        # Tool calls are a successful model response even when content is empty.
        if text or tool_calls:
            return {
                "provider": provider,
                "model": model,
                "content": text or "",
                "tool_calls": tool_calls or [],
                "finish_reason": finish_reason or (
                    "tool_calls" if tool_calls else "stop"
                ),
            }

        if status == "auth_error":
            print(f"  [STOP] {provider}: authentication failed")
            break

        if status == "payment_required":
            print(f"  [STOP] {provider}: payment/quota failure")
            break

        if status in ("missing_key", "missing_account_id"):
            break

        if primary_only:
            break

        print(f"  [FALLBACK] {provider} -> next model")

    return None



# ============================================================
# ROUTER
# ============================================================

def route(
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[list[dict]] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Any = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> dict:
    """
    Route either a legacy prompt or a full OpenAI-compatible request.

    If `messages` is supplied, it is preserved end-to-end, including:
    - system/user/assistant/tool messages
    - assistant tool_calls
    - tool results
    """
    print("\n==================================================")
    print(" AI ROUTER v7")
    print("==================================================")

    if provider_override:
        providers = [provider_override]
    else:
        providers = PROVIDER_ORDER

    overall_start = time.perf_counter()

    for provider in providers:
        if provider not in CONFIG:
            print(f"\n[ERROR] Unknown provider: {provider}")
            continue

        print(f"\n[TRY] {provider}")

        result = call_provider(
            provider=provider,
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            model_override=model_override,
        )

        if result:
            total_time = time.perf_counter() - overall_start
            result["elapsed"] = total_time

            print("\n==================================================")
            print(
                f" SUCCESS: {result['provider']}/{result['model']}"
            )
            print(f" TOTAL TIME: {total_time:.2f}s")
            if result.get("tool_calls"):
                print(
                    f" TOOL CALLS: {len(result['tool_calls'])}"
                )
            print("==================================================")
            return result

    total_time = time.perf_counter() - overall_start

    return {
        "provider": None,
        "model": None,
        "content": (
            "All configured providers and "
            "fallback models were exhausted."
        ),
        "tool_calls": [],
        "finish_reason": "error",
        "elapsed": total_time,
    }



# ============================================================
# TEST ALL PRIMARY MODELS
# ============================================================

def test_all() -> None:

    print(
        "\n=================================================="
    )

    print(
        " PROVIDER PRIMARY MODEL TEST"
    )

    print(
        "=================================================="
    )

    print(
        f"Test prompt: {TEST_PROMPT}"
    )

    print(
        f"Expected response: {TEST_EXPECTED}"
    )

    print(
        "Only PRIMARY models will be tested."
    )

    print(
        "Fallback models will NOT be consumed."
    )

    print(
        "--------------------------------------------------"
    )

    results = []

    total_start = time.perf_counter()

    for provider in PROVIDER_ORDER:

        cfg = CONFIG[provider]

        primary = cfg["model"]

        print(
            f"\n[TEST] {provider}"
        )

        print(
            "--------------------------------------------------"
        )

        provider_start = time.perf_counter()

        # ----------------------------------------------------
        # Credential check
        # ----------------------------------------------------

        if not provider_has_credentials(provider):

            print_provider_skip(provider)

            results.append(
                {
                    "provider": provider,
                    "status": "SKIPPED",
                    "time": 0.0,
                    "model": primary,
                }
            )

            continue

        # ----------------------------------------------------
        # Test ONLY primary model
        # ----------------------------------------------------

        result = call_provider(
            provider=provider,
            prompt=TEST_PROMPT,
            system_prompt=None,
            model_override=None,
            primary_only=True,
        )

        elapsed = (
            time.perf_counter()
            - provider_start
        )

        # ----------------------------------------------------
        # API/model failure
        # ----------------------------------------------------

        if not result:

            print(
                f"  [TEST-FAIL] "
                f"{provider}/{primary}"
            )

            results.append(
                {
                    "provider": provider,
                    "status": "FAIL",
                    "time": elapsed,
                    "model": primary,
                }
            )

            continue

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        response_text = (
            result.get("content", "")
            .strip()
        )

        # ----------------------------------------------------
        # EXACT RESPONSE VALIDATION
        # ----------------------------------------------------

        if response_text == TEST_EXPECTED:

            print(
                f"  [TEST-PASS] "
                f"{provider}/{primary}"
            )

            print(
                f"  [RESPONSE] "
                f"{response_text}"
            )

            status = "PASS"

        else:

            print(
                f"  [TEST-FAIL] "
                f"{provider}/{primary}"
            )

            print(
                f"  [EXPECTED] "
                f"{TEST_EXPECTED}"
            )

            print(
                f"  [RECEIVED] "
                f"{response_text[:500]}"
            )

            status = "FAIL"

        results.append(
            {
                "provider": provider,
                "status": status,
                "time": elapsed,
                "model": primary,
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n=================================================="
    )

    print(
        " TEST SUMMARY"
    )

    print(
        "=================================================="
    )

    print(
        f"{'PROVIDER':15} "
        f"{'STATUS':10} "
        f"{'TIME':10} "
        f"MODEL"
    )

    print(
        "-" * 90
    )

    passed = 0
    failed = 0
    skipped = 0

    for item in results:

        status = item["status"]

        if status == "PASS":
            passed += 1

        elif status == "FAIL":
            failed += 1

        elif status == "SKIPPED":
            skipped += 1

        print(
            f"{item['provider']:15} "
            f"{status:10} "
            f"{item['time']:7.2f}s "
            f"{item['model']}"
        )

    print(
        "\n--------------------------------------------------"
    )

    print(
        f"PASS   : {passed}"
    )

    print(
        f"FAIL   : {failed}"
    )

    print(
        f"SKIP   : {skipped}"
    )

    print(
        f"TOTAL  : {len(results)}"
    )

    print(
        f"TIME   : {total_elapsed:.2f}s"
    )

    print(
        "=================================================="
    )

# ============================================================
# PROVIDER MODEL DISCOVERY
# ============================================================

def discover_models(
    provider: str,
) -> None:
    """
    Query the provider's model endpoint where supported.
    """

    cfg = CONFIG[provider]

    if not provider_has_credentials(
        provider
    ):

        print_provider_skip(
            provider
        )

        return

    print(
        f"\n[DISCOVER] {provider}"
    )

    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    if provider == "ollama":

        url = (
            f"{cfg['base_url']}/api/tags"
        )

        try:

            response = requests.get(
                url,
                timeout=10,
            )

            print(
                f"  HTTP "
                f"{response.status_code}"
            )

            if response.status_code != 200:

                print(
                    response.text[:1000]
                )

                return

            data = response.json()

            models = data.get(
                "models",
                [],
            )

            if not models:

                print(
                    "  [NONE] No Ollama models installed."
                )

                return

            for item in models:

                name = item.get(
                    "name",
                    "",
                )

                if name:
                    print(
                        f"  - {name}"
                    )

            return

        except requests.RequestException as exc:

            print(
                f"  [NETWORK-ERROR] "
                f"{type(exc).__name__}: {exc}"
            )

            return

    # --------------------------------------------------------
    # Cloudflare
    # --------------------------------------------------------

    if provider == "cloudflare":

        print(
            "  [INFO] Cloudflare Workers AI "
            "does not use the same /models "
            "endpoint as Groq."
        )

        print(
            "  [INFO] Configured Workers AI "
            "models are shown by --models."
        )

        return

    # --------------------------------------------------------
    # Google
    # --------------------------------------------------------

    if provider == "google":

        url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models"
        )

        headers = {
            "x-goog-api-key":
                cfg["api_key"],
        }

    # --------------------------------------------------------
    # OpenAI-compatible providers
    # --------------------------------------------------------

    else:

        url = (
            f"{cfg['base_url']}/models"
        )

        headers = {
            "Authorization":
                f"Bearer {cfg['api_key']}",
        }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )

        print(
            f"  HTTP "
            f"{response.status_code}"
        )

        if response.status_code != 200:

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return

        data = response.json()

        # ----------------------------------------------------
        # Google
        # ----------------------------------------------------

        if provider == "google":

            models = data.get(
                "models",
                [],
            )

            for item in models:

                name = item.get(
                    "name",
                    "",
                )

                if name:
                    print(
                        f"  - {name}"
                    )

            return

        # ----------------------------------------------------
        # OpenAI-compatible
        # ----------------------------------------------------

        models = data.get(
            "data",
            [],
        )

        if not models:

            print(
                "  [NONE]"
            )

            return

        for item in models:

            model_id = item.get(
                "id",
                "",
            )

            if model_id:
                print(
                    f"  - {model_id}"
                )

    except requests.RequestException as exc:

        print(
            f"  [NETWORK-ERROR] "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# STATUS
# ============================================================

def show_status() -> None:

    print(
        "\n=================================================="
    )

    print(
        " PROVIDER STATUS"
    )

    print(
        "=================================================="
    )

    for provider in PROVIDER_ORDER:

        cfg = CONFIG[provider]

        if provider == "ollama":

            status = "LOCAL"

        elif provider == "cloudflare":

            if (
                cfg.get("api_key")
                and cfg.get("account_id")
            ):

                status = "READY"

            elif cfg.get("api_key"):

                status = (
                    "MISSING ACCOUNT ID"
                )

            else:

                status = "NO API KEY"

        else:

            status = (
                "READY"
                if cfg.get("api_key")
                else "NO API KEY"
            )

        print(
            f"{provider:15} "
            f"{status:20} "
            f"primary={cfg['model']}"
        )

        fallbacks = cfg.get(
            "fallback_models",
            [],
        )

        print(
            f"{'':15} "
            f"fallbacks={len(fallbacks)}"
        )

    print(
        "==================================================\n"
    )


# ============================================================
# MODELS
# ============================================================

def show_models() -> None:

    print(
        "\n=================================================="
    )

    print(
        " CONFIGURED MODELS"
    )

    print(
        "=================================================="
    )

    for provider in PROVIDER_ORDER:

        cfg = CONFIG[provider]

        models = unique_models(
            cfg["model"],
            cfg.get(
                "fallback_models",
                [],
            ),
        )

        print(
            f"\n[{provider}]"
        )

        for index, model in enumerate(
            models,
            start=1,
        ):

            label = (
                "PRIMARY"
                if index == 1
                else "FALLBACK"
            )

            print(
                f"  {index:2}. "
                f"{label:8} "
                f"{model}"
            )

    print(
        "\n=================================================="
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Multi-provider AI token router"
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive chat mode",
    )

    parser.add_argument(
        "--system",
        "-s",
        help="System prompt",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show provider configuration status",
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="Show configured models",
    )

    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test primary model of every configured provider",
    )

    parser.add_argument(
        "--discover",
        metavar="PROVIDER",
        help=(
            "Discover models available from a provider"
        ),
    )

    parser.add_argument(
        "--provider",
        help=(
            "Use only this provider "
            "instead of normal routing"
        ),
    )

    parser.add_argument(
        "--model",
        help=(
            "Use only this model. "
            "Requires --provider."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # PROVIDER VALIDATION
    # ========================================================

    if args.provider:

        if args.provider not in CONFIG:

            print(
                f"[ERROR] Unknown provider: "
                f"{args.provider}"
            )

            print(
                "\nAvailable providers:"
            )

            for provider in CONFIG:

                print(
                    f"  - {provider}"
                )

            return

    # ========================================================
    # DISCOVERY VALIDATION
    # ========================================================

    if args.discover:

        if args.discover not in CONFIG:

            print(
                f"[ERROR] Unknown provider: "
                f"{args.discover}"
            )

            print(
                "\nAvailable providers:"
            )

            for provider in CONFIG:

                print(
                    f"  - {provider}"
                )

            return

        discover_models(
            args.discover
        )

        return

    # ========================================================
    # MODEL VALIDATION
    # ========================================================

    if args.model and not args.provider:

        print(
            "[ERROR] --model requires --provider"
        )

        print(
            "\nExample:"
        )

        print(
            "python token_router.py "
            "--provider huggingface "
            "--model Qwen/Qwen3-Coder-Next "
            "\"Hello\""
        )

        return

    # ========================================================
    # STATUS
    # ========================================================

    if args.status:

        show_status()

        return

    # ========================================================
    # MODELS
    # ========================================================

    if args.models:

        show_models()

        return

    # ========================================================
    # TEST ALL
    # ========================================================

    if args.test_all:

        test_all()

        return

    # ========================================================
    # INTERACTIVE
    # ========================================================

    if args.interactive:

        print(
            "=== Multi-Provider AI Router v7 ==="
        )

        system_prompt = (
            args.system
            or "You are a helpful assistant."
        )

        print(
            f"System: {system_prompt}"
        )

        if args.provider:

            print(
                f"Provider: "
                f"{args.provider}"
            )

        if args.model:

            print(
                f"Model: "
                f"{args.model}"
            )

        while True:

            try:

                user_input = input(
                    "\nYou: "
                ).strip()

                if user_input.lower() in (
                    "exit",
                    "quit",
                    "/exit",
                    "/quit",
                ):

                    break

                if not user_input:

                    continue

                result = route(
                    prompt=user_input,
                    system_prompt=system_prompt,
                    provider_override=
                        args.provider,
                    model_override=
                        args.model,
                )

                if result["provider"]:

                    print(
                        f"\n["
                        f"{result['provider']}/"
                        f"{result['model']}"
                        f"]"
                    )

                    print(
                        result["content"]
                    )

                else:

                    print(
                        "\n[ERROR]: "
                        f"{result['content']}"
                    )

            except KeyboardInterrupt:

                print(
                    "\nExiting."
                )

                break

        return

    # ========================================================
    # ONE SHOT
    # ========================================================

    if args.prompt:

        print(
            f"Query: {args.prompt}"
        )

        result = route(
            prompt=args.prompt,
            system_prompt=args.system,
            provider_override=
                args.provider,
            model_override=
                args.model,
        )

        if result["provider"]:

            print(
                f"\n["
                f"{result['provider']}/"
                f"{result['model']}"
                f"]:"
            )

            print(
                result["content"]
            )

        else:

            print(
                "\n[ERROR]: "
                f"{result['content']}"
            )

        return

    # ========================================================
    # HELP
    # ========================================================

    parser.print_help()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()