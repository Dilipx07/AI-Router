"""
Provider abstraction layer for AI Router v1.

Every provider adapter must conform to this interface.
The router itself should not need to know provider-specific
HTTP/API implementation details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# NORMALIZED USAGE
# ============================================================

@dataclass(frozen=True)
class ProviderUsage:
    """
    Normalized token usage reported by a provider.
    """

    prompt_tokens: int = 0

    completion_tokens: int = 0

    total_tokens: int = 0

    @property
    def known(self) -> bool:
        """
        Whether the provider supplied meaningful usage data.
        """
        return (
            self.prompt_tokens > 0
            or self.completion_tokens > 0
            or self.total_tokens > 0
        )

    def normalized(self) -> "ProviderUsage":
        """
        Return usage with total_tokens guaranteed to be
        consistent when enough information is available.
        """

        prompt = max(0, int(self.prompt_tokens))
        completion = max(0, int(self.completion_tokens))
        total = max(0, int(self.total_tokens))

        if total == 0 and (
            prompt > 0
            or completion > 0
        ):
            total = prompt + completion

        return ProviderUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )


# ============================================================
# NORMALIZED PROVIDER RESULT
# ============================================================

@dataclass
class ProviderResult:
    """
    Normalized result returned by every provider.

    The router operates exclusively on this structure instead
    of dealing with provider-specific response formats.
    """

    success: bool = False

    provider: str = ""
    model: str = ""

    content: str = ""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    finish_reason: Optional[str] = None

    status: str = "error"

    http_status: Optional[int] = None

    elapsed: float = 0.0

    error: Optional[str] = None

    raw_response: Optional[dict[str, Any]] = None

    usage: ProviderUsage = field(
        default_factory=ProviderUsage
    )
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success_result(
        cls,
        *,
        provider: str,
        model: str,
        content: str = "",
        tool_calls: Optional[list[dict[str, Any]]] = None,
        finish_reason: Optional[str] = None,
        elapsed: float = 0.0,
        http_status: Optional[int] = None,
        raw_response: Optional[dict[str, Any]] = None,
        usage: Optional[ProviderUsage] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult":
        return cls(
            success=True,
            provider=provider,
            model=model,
            content=content,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            status="success",
            http_status=http_status,
            elapsed=elapsed,
            raw_response=raw_response,
            usage=(
                usage.normalized()
                if usage is not None
                else ProviderUsage()
            ),
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        *,
        provider: str,
        model: str,
        status: str,
        error: Optional[str] = None,
        elapsed: float = 0.0,
        http_status: Optional[int] = None,
        raw_response: Optional[dict[str, Any]] = None,
        usage: Optional[ProviderUsage] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ProviderResult":
        return cls(
            success=False,
            provider=provider,
            model=model,
            status=status,
            error=error,
            elapsed=elapsed,
            http_status=http_status,
            raw_response=raw_response,
            usage=(
                usage.normalized()
                if usage is not None
                else ProviderUsage()
            ),
            metadata=metadata or {},
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip())

    @property
    def usable(self) -> bool:
        """
        Whether this result can be returned to the caller.

        A tool-call response is valid even when textual content
        is empty.
        """
        return self.success and (
            self.has_content or self.has_tool_calls
        )


# ============================================================
# PROVIDER REQUEST
# ============================================================

@dataclass
class ProviderRequest:
    """
    Provider-independent representation of an AI request.

    Provider adapters receive this object and translate it into
    their native API format where necessary.
    """

    messages: list[dict[str, Any]]

    tools: Optional[list[dict[str, Any]]] = None

    tool_choice: Any = None

    max_tokens: Optional[int] = None

    temperature: Optional[float] = None
    
    top_p: Optional[float] = None

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# PROVIDER CAPABILITIES
# ============================================================

@dataclass(frozen=True)
class ProviderCapabilities:
    """
    Declares capabilities supported by a provider.
    """

    chat: bool = True

    streaming: bool = False

    tools: bool = False

    structured_output: bool = False

    vision: bool = False

    embeddings: bool = False

    model_discovery: bool = False


# ============================================================
# PROVIDER BASE CLASS
# ============================================================

class BaseProvider(ABC):
    """
    Abstract provider interface.

    Provider implementations must inherit from this class.

    The router should communicate with providers exclusively
    through this interface.
    """

    name: str = ""

    capabilities = ProviderCapabilities()

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def __init__(
        self,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}

    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Return the canonical provider identifier."""
        return self.name

    # --------------------------------------------------------
    # CREDENTIALS
    # --------------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True when this provider is configured and usable.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        model: str,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute a chat request against the provider.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # MODEL DISCOVERY
    # --------------------------------------------------------

    def discover_models(self) -> list[str]:
        """
        Return models exposed by the provider.

        Providers that do not support discovery may return an
        empty list.
        """
        return []

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    def health_check(self) -> bool:
        """
        Lightweight provider health check.

        The default implementation uses availability. Concrete
        providers can override this with an actual API health
        check.
        """
        return self.is_available()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def supports(self, capability: str) -> bool:
        """
        Check whether a provider advertises a capability.
        """

        return bool(
            getattr(
                self.capabilities,
                capability,
                False,
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.provider_name!r})"
        )