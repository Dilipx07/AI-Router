#!/usr/bin/env python3
"""
token_router.py v1

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
import os
import time
from typing import Optional

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

        # Primary cloud model
        "model":
            "minimax-m3:cloud",

        # Local fallbacks
        "fallback_models": [

            "qwen2.5-coder:1.5b",

            "qwen2.5-coder:0.5b",

            "gemma3:1b",

        ],
    },
}


# ============================================================
# PROVIDER ORDER
# ============================================================

PROVIDER_ORDER = [
    "huggingface",
    "google",
    "groq",
    "openrouter",
    "cloudflare",
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
    prompt: str,
    system_prompt: Optional[str] = None,
) -> list:

    messages = []

    if system_prompt:

        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return messages


def print_request_time(
    elapsed: float,
) -> None:

    print(
        f"  [TIME] {elapsed:.2f}s"
    )


def extract_openai_content(
    data: dict,
) -> str:
    """
    Extract assistant text from OpenAI-compatible
    responses.

    Supports:

        choices[0].message.content

    where content can be either:

        string

    or:

        [
            {"type": "text", "text": "..."}
        ]
    """

    choices = data.get(
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

    if isinstance(content, str):

        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = item.get(
                "text",
                "",
            )

            if text:
                parts.append(
                    str(text)
                )

        return "".join(parts).strip()

    return str(
        content or ""
    ).strip()


# ============================================================
# GOOGLE GEMINI
# ============================================================

def call_google(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], str]:

    cfg = CONFIG["google"]

    url = (
        f"{cfg['base_url']}/"
        f"{model}:generateContent"
    )

    headers = {
        "Content-Type":
            "application/json",

        "x-goog-api-key":
            cfg["api_key"],
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }

    if system_prompt:

        payload["systemInstruction"] = {
            "parts": [
                {
                    "text":
                        system_prompt,
                }
            ]
        }

    start = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [DEBUG] google/{model} "
            f"HTTP {response.status_code}"
        )

        print_request_time(
            elapsed
        )

        if response.status_code == 200:

            data = response.json()

            candidates = data.get(
                "candidates",
                [],
            )

            if not candidates:

                print(
                    "  [WARN] Google returned 200 "
                    "but no candidates"
                )

                print(
                    f"  [DEBUG] "
                    f"{response.text[:1000]}"
                )

                return None, "empty"

            content = candidates[0].get(
                "content",
                {},
            )

            parts = content.get(
                "parts",
                [],
            )

            text_parts = []

            for part in parts:

                if not isinstance(
                    part,
                    dict,
                ):
                    continue

                text = part.get(
                    "text",
                    "",
                )

                if text:
                    text_parts.append(
                        text
                    )

            text = "".join(
                text_parts
            ).strip()

            if not text:

                return None, "empty"

            print(
                f"  [OK] google/{model}"
            )

            return text, "success"

        if response.status_code == 429:

            print(
                f"  [RATE-LIMIT] "
                f"google/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "rate_limit"

        if response.status_code in (
            401,
            403,
        ):

            print(
                f"  [AUTH-ERROR] "
                f"google/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "auth_error"

        if response.status_code == 404:

            print(
                f"  [MODEL-NOT-FOUND] "
                f"google/{model}"
            )

            return None, "model_not_found"

        print(
            f"  [FAIL] google/{model}: "
            f"HTTP {response.status_code}"
        )

        print(
            f"  [DEBUG] "
            f"{response.text[:1000]}"
        )

        return None, "error"

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [NETWORK-ERROR] "
            f"google/{model}: "
            f"{type(exc).__name__}: {exc}"
        )

        print_request_time(
            elapsed
        )

        return None, "network_error"


# ============================================================
# OLLAMA NATIVE API
# ============================================================

def call_ollama(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], str]:
    """
    Call Ollama using its native /api/chat endpoint.

    This supports both:
    - local Ollama models
    - Ollama cloud models such as minimax-m3:cloud
    """

    cfg = CONFIG["ollama"]

    url = (
        f"{cfg['base_url']}"
        f"/api/chat"
    )

    headers = {
        "Content-Type":
            "application/json",
    }

    messages = build_messages(
        prompt,
        system_prompt,
    )

    payload = {
        "model":
            model,

        "messages":
            messages,

        "stream":
            False,

        "options": {
            "num_predict":
                DEFAULT_MAX_TOKENS,
        },
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [DEBUG] "
            f"ollama/{model} "
            f"HTTP {response.status_code}"
        )

        print_request_time(
            elapsed
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if response.status_code == 200:

            data = response.json()

            message = data.get(
                "message",
                {},
            )

            text = message.get(
                "content",
                "",
            )

            if isinstance(text, str):

                text = text.strip()

            else:

                text = str(
                    text or ""
                ).strip()

            if not text:

                print(
                    f"  [WARN] "
                    f"ollama/{model}: "
                    "empty response"
                )

                print(
                    f"  [DEBUG] "
                    f"{response.text[:1500]}"
                )

                return None, "empty"

            print(
                f"  [OK] "
                f"ollama/{model}"
            )

            return text, "success"

        # ----------------------------------------------------
        # MODEL NOT FOUND
        # ----------------------------------------------------

        if response.status_code == 404:

            print(
                f"  [MODEL-NOT-FOUND] "
                f"ollama/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            print(
                "  [OLLAMA] "
                "Model is probably not installed "
                "or unavailable."
            )

            print(
                "  [OLLAMA] "
                "Run: ollama list"
            )

            print(
                "  [OLLAMA] "
                f"Expected model: {model}"
            )

            return None, "model_not_found"

        # ----------------------------------------------------
        # AUTH
        # ----------------------------------------------------

        if response.status_code in (
            401,
            403,
        ):

            print(
                f"  [AUTH-ERROR] "
                f"ollama/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "auth_error"

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:

            print(
                f"  [RATE-LIMIT] "
                f"ollama/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "rate_limit"

        # ----------------------------------------------------
        # PAYMENT / CLOUD QUOTA
        # ----------------------------------------------------

        if response.status_code == 402:

            print(
                f"  [PAYMENT/QUOTA] "
                f"ollama/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "payment_required"

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        print(
            f"  [FAIL] "
            f"ollama/{model}: "
            f"HTTP {response.status_code}"
        )

        print(
            f"  [DEBUG] "
            f"{response.text[:1500]}"
        )

        return None, "error"

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [NETWORK-ERROR] "
            f"ollama/{model}: "
            f"{type(exc).__name__}: {exc}"
        )

        print_request_time(
            elapsed
        )

        return None, "network_error"


# ============================================================
# OPENAI-COMPATIBLE PROVIDERS
# ============================================================

def call_openai_compatible(
    provider: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], str]:

    cfg = CONFIG[provider]

    url = (
        f"{cfg['base_url']}"
        f"/chat/completions"
    )

    headers = {
        "Content-Type":
            "application/json",
    }

    if provider != "ollama":

        headers["Authorization"] = (
            f"Bearer {cfg['api_key']}"
        )

    payload = {
        "model":
            model,

        "messages":
            build_messages(
                prompt,
                system_prompt,
            ),

        "max_tokens":
            DEFAULT_MAX_TOKENS,
    }

    start = time.perf_counter()

    try:

        timeout = (
            OLLAMA_TIMEOUT
            if provider == "ollama"
            else DEFAULT_TIMEOUT
        )

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [DEBUG] "
            f"{provider}/{model} "
            f"HTTP {response.status_code}"
        )

        print_request_time(
            elapsed
        )

        if response.status_code == 200:

            data = response.json()

            text = extract_openai_content(
                data
            )

            if not text:

                print(
                    f"  [WARN] "
                    f"{provider}/{model}: "
                    "empty response"
                )

                print(
                    f"  [DEBUG] "
                    f"{response.text[:1500]}"
                )

                return None, "empty"

            print(
                f"  [OK] "
                f"{provider}/{model}"
            )

            return text, "success"

        if response.status_code == 429:

            print(
                f"  [RATE-LIMIT] "
                f"{provider}/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "rate_limit"

        if response.status_code in (
            401,
            403,
        ):

            print(
                f"  [AUTH-ERROR] "
                f"{provider}/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "auth_error"

        if response.status_code == 404:

            print(
                f"  [MODEL-NOT-FOUND] "
                f"{provider}/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            if provider == "ollama":

                print(
                    "  [OLLAMA] "
                    "Model is probably not installed."
                )

                print(
                    "  [OLLAMA] "
                    "Run: ollama list"
                )

                print(
                    "  [OLLAMA] "
                    f"Then install with: "
                    f"ollama pull {model}"
                )

            return None, "model_not_found"

        if response.status_code == 402:

            print(
                f"  [PAYMENT/QUOTA] "
                f"{provider}/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "payment_required"

        print(
            f"  [FAIL] "
            f"{provider}/{model}: "
            f"HTTP {response.status_code}"
        )

        print(
            f"  [DEBUG] "
            f"{response.text[:1500]}"
        )

        return None, "error"

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [NETWORK-ERROR] "
            f"{provider}/{model}: "
            f"{type(exc).__name__}: {exc}"
        )

        print_request_time(
            elapsed
        )

        return None, "network_error"


# ============================================================
# HUGGING FACE
# ============================================================

def call_huggingface(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], str]:

    cfg = CONFIG["huggingface"]

    url = (
        f"{cfg['base_url']}"
        f"/chat/completions"
    )

    headers = {
        "Authorization":
            f"Bearer {cfg['api_key']}",

        "Content-Type":
            "application/json",
    }

    payload = {
        "model":
            model,

        "messages":
            build_messages(
                prompt,
                system_prompt,
            ),

        "max_tokens":
            DEFAULT_MAX_TOKENS,
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=HUGGINGFACE_TIMEOUT,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [DEBUG] "
            f"huggingface/{model} "
            f"HTTP {response.status_code}"
        )

        print_request_time(
            elapsed
        )

        if response.status_code == 200:

            data = response.json()

            text = extract_openai_content(
                data
            )

            if not text:

                print(
                    f"  [WARN] "
                    f"huggingface/{model}: "
                    "empty response"
                )

                print(
                    f"  [DEBUG] "
                    f"{response.text[:1500]}"
                )

                return None, "empty"

            print(
                f"  [OK] "
                f"huggingface/{model}"
            )

            return text, "success"

        if response.status_code == 429:

            print(
                f"  [RATE-LIMIT] "
                f"huggingface/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "rate_limit"

        if response.status_code in (
            401,
            403,
        ):

            print(
                f"  [AUTH-ERROR] "
                f"huggingface/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "auth_error"

        if response.status_code == 404:

            print(
                f"  [MODEL-NOT-FOUND] "
                f"huggingface/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "model_not_found"

        if response.status_code == 402:

            print(
                f"  [PAYMENT/QUOTA] "
                f"huggingface/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "payment_required"

        print(
            f"  [FAIL] "
            f"huggingface/{model}: "
            f"HTTP {response.status_code}"
        )

        print(
            f"  [DEBUG] "
            f"{response.text[:1500]}"
        )

        return None, "error"

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [NETWORK-ERROR] "
            f"huggingface/{model}: "
            f"{type(exc).__name__}: {exc}"
        )

        print_request_time(
            elapsed
        )

        return None, "network_error"


# ============================================================
# CLOUDFLARE
# ============================================================

def call_cloudflare(
    model: str,
    prompt: str,
    system_prompt: Optional[str],
) -> tuple[Optional[str], str]:

    cfg = CONFIG["cloudflare"]

    if not cfg.get("api_key"):

        return None, "missing_key"

    if not cfg.get("account_id"):

        return None, "missing_account_id"

    # --------------------------------------------------------
    # Use Cloudflare's OpenAI-compatible endpoint.
    #
    # This avoids having to support multiple response formats
    # from /ai/run.
    # --------------------------------------------------------

    url = (
        f"{cfg['base_url']}/"
        f"{cfg['account_id']}/"
        f"ai/v1/chat/completions"
    )

    headers = {
        "Authorization":
            f"Bearer {cfg['api_key']}",

        "Content-Type":
            "application/json",
    }

    payload = {
        "model":
            model,

        "messages":
            build_messages(
                prompt,
                system_prompt,
            ),

        "max_tokens":
            DEFAULT_MAX_TOKENS,
    }

    start = time.perf_counter()

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [DEBUG] "
            f"cloudflare/{model} "
            f"HTTP {response.status_code}"
        )

        print_request_time(
            elapsed
        )

        if response.status_code == 200:

            data = response.json()

            text = extract_openai_content(
                data
            )

            if not text:

                print(
                    f"  [WARN] "
                    f"cloudflare/{model}: "
                    "empty response"
                )

                print(
                    f"  [DEBUG] "
                    f"{response.text[:2000]}"
                )

                return None, "empty"

            print(
                f"  [OK] "
                f"cloudflare/{model}"
            )

            return text, "success"

        if response.status_code == 429:

            print(
                f"  [RATE-LIMIT] "
                f"cloudflare/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1000]}"
            )

            return None, "rate_limit"

        if response.status_code in (
            401,
            403,
        ):

            print(
                f"  [AUTH-ERROR] "
                f"cloudflare/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "auth_error"

        if response.status_code == 404:

            print(
                f"  [MODEL-NOT-FOUND] "
                f"cloudflare/{model}"
            )

            print(
                f"  [DEBUG] "
                f"{response.text[:1500]}"
            )

            return None, "model_not_found"

        print(
            f"  [FAIL] "
            f"cloudflare/{model}: "
            f"HTTP {response.status_code}"
        )

        print(
            f"  [DEBUG] "
            f"{response.text[:1500]}"
        )

        return None, "error"

    except requests.RequestException as exc:

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"  [NETWORK-ERROR] "
            f"cloudflare/{model}: "
            f"{type(exc).__name__}: {exc}"
        )

        print_request_time(
            elapsed
        )

        return None, "network_error"


# ============================================================
# PROVIDER CALL
# ============================================================

def call_provider(
    provider: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    model_override: Optional[str] = None,
    primary_only: bool = False,
) -> Optional[dict]:

    cfg = CONFIG[provider]

    if not provider_has_credentials(
        provider
    ):

        print_provider_skip(
            provider
        )

        return None

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    if model_override:

        models = [
            model_override
        ]

    elif primary_only:

        models = [
            cfg["model"]
        ]

    else:

        models = unique_models(
            cfg["model"],
            cfg.get(
                "fallback_models",
                [],
            ),
        )

    print(
        f"  [MODELS] "
        f"{len(models)} configured"
    )

    # --------------------------------------------------------
    # Try models
    # --------------------------------------------------------

    for index, model in enumerate(
        models,
        start=1,
    ):

        print(
            f"\n  [MODEL {index}/{len(models)}] "
            f"{provider}/{model}"
        )

        if provider == "google":

            text, status = call_google(
                model,
                prompt,
                system_prompt,
            )

        elif provider == "huggingface":

            text, status = call_huggingface(
                model,
                prompt,
                system_prompt,
            )

        elif provider == "cloudflare":

            text, status = call_cloudflare(
                model,
                prompt,
                system_prompt,
            )

        elif provider == "ollama":

            text, status = call_ollama(
                model,
                prompt,
                system_prompt,
            )

        else:

            text, status = (
                call_openai_compatible(
                    provider,
                    model,
                    prompt,
                    system_prompt,
                )
            )

        if text:

            return {
                "provider":
                    provider,

                "model":
                    model,

                "content":
                    text,
            }

        # ----------------------------------------------------
        # Stop provider for fatal credential errors.
        # ----------------------------------------------------

        if status == "auth_error":

            print(
                f"  [STOP] {provider}: "
                "authentication failed"
            )

            break

        if status == "payment_required":

            print(
                f"  [STOP] {provider}: "
                "payment/quota failure"
            )

            break

        if status in (
            "missing_key",
            "missing_account_id",
        ):

            break

        if primary_only:

            break

        print(
            f"  [FALLBACK] "
            f"{provider} -> next model"
        )

    return None


# ============================================================
# ROUTER
# ============================================================

def route(
    prompt: str,
    system_prompt: Optional[str] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> dict:

    print(
        "\n=================================================="
    )

    print(
        " AI ROUTER v1"
    )

    print(
        "=================================================="
    )

    if provider_override:

        providers = [
            provider_override
        ]

    else:

        providers = PROVIDER_ORDER

    overall_start = time.perf_counter()

    for provider in providers:

        if provider not in CONFIG:

            print(
                f"\n[ERROR] Unknown provider: "
                f"{provider}"
            )

            continue

        print(
            f"\n[TRY] {provider}"
        )

        result = call_provider(
            provider=provider,
            prompt=prompt,
            system_prompt=system_prompt,
            model_override=model_override,
        )

        if result:

            total_time = (
                time.perf_counter()
                - overall_start
            )

            result["elapsed"] = (
                total_time
            )

            print(
                "\n=================================================="
            )

            print(
                f" SUCCESS: "
                f"{result['provider']}/"
                f"{result['model']}"
            )

            print(
                f" TOTAL TIME: "
                f"{total_time:.2f}s"
            )

            print(
                "=================================================="
            )

            return result

    total_time = (
        time.perf_counter()
        - overall_start
    )

    return {
        "provider":
            None,

        "model":
            None,

        "content":
            (
                "All configured providers and "
                "fallback models were exhausted."
            ),

        "elapsed":
            total_time,
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
            "=== Multi-Provider AI Router v1 ==="
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