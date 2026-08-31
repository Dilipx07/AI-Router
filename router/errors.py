"""
AI Router v1
Normalized provider errors.

This module defines the router's internal error taxonomy and
normalizes provider/API failures into consistent RouterError
objects.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from providers.base import ProviderUsage

# ============================================================
# ERROR TYPES
# ============================================================

class ErrorType(str, Enum):

    AUTH_FAILURE = "auth_failure"

    RATE_LIMIT = "rate_limit"

    QUOTA_EXCEEDED = "quota_exceeded"

    PAYMENT_REQUIRED = "payment_required"

    MODEL_NOT_FOUND = "model_not_found"

    MODEL_UNAVAILABLE = "model_unavailable"

    CONTEXT_TOO_LARGE = "context_too_large"

    INVALID_REQUEST = "invalid_request"

    TOOL_UNSUPPORTED = "tool_unsupported"

    CONTENT_FILTER = "content_filter"

    TIMEOUT = "timeout"

    NETWORK_FAILURE = "network_failure"

    PROVIDER_FAILURE = "provider_failure"

    EMPTY_RESPONSE = "empty_response"

    UNKNOWN = "unknown"


# ============================================================
# ROUTER ERROR
# ============================================================

class RouterError(Exception):
    """
    Normalized exception used internally by the router.
    """

    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        retryable: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        usage: Optional[ProviderUsage] = None,
    ) -> None:

        super().__init__(message)

        self.error_type = error_type
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable
        self.usage = (
            usage
            if usage is not None
            else ProviderUsage()
        )
        self.metadata = metadata or {}

    def __str__(self) -> str:

        location = ""

        if self.provider:
            location += self.provider

        if self.model:
            location += f"/{self.model}"

        if location:
            location = f" [{location}]"

        return (
            f"{self.error_type.value}"
            f"{location}: "
            f"{self.message}"
        )


# ============================================================
# ERROR TEXT HELPERS
# ============================================================

def _normalize_error_text(
    message: str,
) -> str:
    """
    Normalize provider error text for classification.

    Classification is intentionally conservative and based on
    recognizable provider terminology.
    """

    return " ".join(
        str(message)
        .lower()
        .strip()
        .split()
    )


def _contains_any(
    text: str,
    phrases: tuple[str, ...],
) -> bool:
    """Return True when any phrase occurs in the text."""

    return any(
        phrase in text
        for phrase in phrases
    )


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_http_error(
    status_code: int,
    message: str = "",
) -> ErrorType:
    """
    Classify an HTTP/API failure into the router's normalized
    ErrorType taxonomy.

    Status code is the primary signal. Error text is used to
    distinguish provider-specific quota, billing, model, and
    capability failures where the HTTP status alone is
    insufficient.
    """

    text = _normalize_error_text(message)

    # --------------------------------------------------------
    # AUTHENTICATION / AUTHORIZATION
    # --------------------------------------------------------

    if status_code == 401:

        return ErrorType.AUTH_FAILURE

    if status_code == 403:

        if _contains_any(
            text,
            (
                "quota",
                "quota exceeded",
                "quota exhausted",
                "usage limit",
                "usage exceeded",
                "credits exhausted",
                "credits exceeded",
                "insufficient credits",
            ),
        ):
            return ErrorType.QUOTA_EXCEEDED

        if _contains_any(
            text,
            (
                "payment required",
                "billing",
                "billing required",
                "subscription required",
                "insufficient balance",
            ),
        ):
            return ErrorType.PAYMENT_REQUIRED

        return ErrorType.AUTH_FAILURE

    # --------------------------------------------------------
    # INVALID REQUEST
    # --------------------------------------------------------

    if status_code == 400:

        if _contains_any(
            text,
            (
                "tool",
                "tools",
                "function calling",
                "tool call",
                "tool use",
                "unsupported tool",
            ),
        ) and _contains_any(
            text,
            (
                "not supported",
                "unsupported",
                "does not support",
            ),
        ):
            return ErrorType.TOOL_UNSUPPORTED

        if _contains_any(
            text,
            (
                "context length",
                "context window",
                "too many tokens",
                "maximum context",
                "max context",
                "token limit",
            ),
        ):
            return ErrorType.CONTEXT_TOO_LARGE

        if _contains_any(
            text,
            (
                "content filter",
                "content policy",
                "safety filter",
                "safety policy",
                "blocked content",
            ),
        ):
            return ErrorType.CONTENT_FILTER

        return ErrorType.INVALID_REQUEST

    if status_code == 422:

        return ErrorType.INVALID_REQUEST

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if status_code == 404:

        if _contains_any(
            text,
            (
                "model",
                "model not found",
                "no such model",
                "unknown model",
                "model does not exist",
            ),
        ):
            return ErrorType.MODEL_NOT_FOUND

        return ErrorType.MODEL_NOT_FOUND

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    if status_code == 408:

        return ErrorType.TIMEOUT

    # --------------------------------------------------------
    # CONFLICT
    # --------------------------------------------------------

    if status_code == 409:

        return ErrorType.PROVIDER_FAILURE

    # --------------------------------------------------------
    # CONTEXT SIZE
    # --------------------------------------------------------

    if status_code == 413:

        return ErrorType.CONTEXT_TOO_LARGE

    # --------------------------------------------------------
    # RATE LIMIT / QUOTA
    # --------------------------------------------------------

    if status_code == 429:

        if _contains_any(
            text,
            (
                "quota",
                "quota exceeded",
                "quota exhausted",
                "daily limit",
                "monthly limit",
                "usage limit",
                "usage exceeded",
                "credits exhausted",
                "credits exceeded",
                "insufficient credits",
                "resource exhausted",
            ),
        ):
            return ErrorType.QUOTA_EXCEEDED

        return ErrorType.RATE_LIMIT

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if status_code == 402:

        return ErrorType.PAYMENT_REQUIRED

    # --------------------------------------------------------
    # SERVER ERRORS
    # --------------------------------------------------------

    if 500 <= status_code <= 599:

        return ErrorType.PROVIDER_FAILURE

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return ErrorType.UNKNOWN


# ============================================================
# HTTP ERROR MAPPING
# ============================================================

def error_from_http_status(
    status_code: int,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    message: str = "",
    metadata: Optional[dict[str, Any]] = None,
) -> RouterError:
    """
    Convert an HTTP/API failure into a normalized RouterError.

    Provider adapters should use this function instead of
    implementing their own HTTP status classification.
    """

    error_type = classify_http_error(
        status_code=status_code,
        message=message,
    )

    retryable = error_type in {
        ErrorType.RATE_LIMIT,
        ErrorType.TIMEOUT,
        ErrorType.NETWORK_FAILURE,
        ErrorType.PROVIDER_FAILURE,
        ErrorType.MODEL_NOT_FOUND,
        ErrorType.MODEL_UNAVAILABLE,
        ErrorType.QUOTA_EXCEEDED,
    }

    # Explicitly non-retryable failures.
    if error_type in {
        ErrorType.AUTH_FAILURE,
        ErrorType.PAYMENT_REQUIRED,
        ErrorType.CONTEXT_TOO_LARGE,
        ErrorType.INVALID_REQUEST,
        ErrorType.TOOL_UNSUPPORTED,
        ErrorType.CONTENT_FILTER,
    }:
        retryable = False

    default_messages = {
        ErrorType.AUTH_FAILURE:
            "Authentication or authorization failed.",

        ErrorType.RATE_LIMIT:
            "Rate limit exceeded.",

        ErrorType.QUOTA_EXCEEDED:
            "Provider quota exceeded.",

        ErrorType.PAYMENT_REQUIRED:
            "Payment or billing is required.",

        ErrorType.MODEL_NOT_FOUND:
            "Model not found.",

        ErrorType.MODEL_UNAVAILABLE:
            "Model is unavailable.",

        ErrorType.CONTEXT_TOO_LARGE:
            "Request exceeds the model context limit.",

        ErrorType.INVALID_REQUEST:
            "Invalid request.",

        ErrorType.TOOL_UNSUPPORTED:
            "Requested tools are not supported.",

        ErrorType.CONTENT_FILTER:
            "Request or response was blocked by a content filter.",

        ErrorType.TIMEOUT:
            "Request timeout.",

        ErrorType.NETWORK_FAILURE:
            "Provider network failure.",

        ErrorType.PROVIDER_FAILURE:
            "Provider server error.",

        ErrorType.EMPTY_RESPONSE:
            "Provider returned an empty response.",

        ErrorType.UNKNOWN:
            f"HTTP {status_code}",
    }

    return RouterError(
        error_type=error_type,
        message=(
            message
            or default_messages[error_type]
        ),
        provider=provider,
        model=model,
        status_code=status_code,
        retryable=retryable,
        metadata=metadata,
    )