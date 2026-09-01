"""
AI Router v1
Provider execution and fallback layer.

The executor is responsible for:
- executing a routed provider request
- resolving provider adapters
- recording runtime health
- performing retry/fallback across ranked candidates
- returning normalized ProviderResult objects

The executor does NOT:
- implement provider-specific HTTP logic
- modify provider/model configuration
- perform candidate scoring
- perform candidate discovery
"""

from __future__ import annotations

from typing import Any, Optional

from providers.base import (
    ProviderRequest,
    ProviderResult,
    ProviderUsage,
)

from .engine import (
    RoutingDecision,
    RoutingRequest,
)
from .errors import (
    ErrorType,
    RouterError,
)
from .ids import create_request_id

# ============================================================
# EXECUTION RESULT
# ============================================================

class ExecutionError(RouterError):
    """
    Final error raised when all routing candidates fail.

    The individual candidate failures are preserved in
    metadata["attempts"].
    """

    pass


# ============================================================
# ROUTER EXECUTOR
# ============================================================

class RouterExecutor:
    """
    Executes provider requests using routing decisions.

    Provider adapters are responsible for communicating with
    external APIs. This class only orchestrates execution,
    health tracking, and fallback.
    """

    def __init__(
        self,
        *,
        providers,
        health_store,
        metrics_store=None,
        quota_store=None,
        router,
    ) -> None:
        """
        Create a router executor.

        Production runtime passes the shared MetricsStore and
        QuotaStore explicitly.

        Older isolated tests may omit them; in that case, create
        local stores so those tests remain independent and do not
        fall back to mutable global state.
        """

        from .metrics import MetricsStore
        from .quota import QuotaStore

        self.providers = providers

        self.health_store = health_store

        self.metrics_store = (
            metrics_store
            if metrics_store is not None
            else MetricsStore()
        )

        self.quota_store = (
            quota_store
            if quota_store is not None
            else QuotaStore()
        )

        self.router = router


    # ========================================================
    # PROVIDER RESOLUTION
    # ========================================================

    def get_provider(
        self,
        provider_name: str,
    ) -> Any:
        """
        Resolve a provider adapter by canonical provider name.
        """

        provider = self.providers.get(
            provider_name
        )

        if provider is None:

            raise RouterError(
                ErrorType.PROVIDER_FAILURE,
                (
                    f"Provider adapter '{provider_name}' "
                    "is not registered."
                ),
                provider=provider_name,
                retryable=False,
            )

        return provider

    # ========================================================
    # SINGLE ATTEMPT
    # ========================================================

    def execute_candidate(
        self,
        candidate: Any,
        request: ProviderRequest,
    ) -> ProviderResult:
        """
        Execute one concrete provider/model candidate.

        Health is updated based on the outcome.
        """

        provider_name = candidate.provider
        model_name = candidate.model

        provider = self.get_provider(
            provider_name
        )

        try:

            result = provider.chat(
                model=model_name,
                request=request,
            )

        except RouterError as error:

            self.health_store.record_failure(
                error
            )
            
            self.metrics_store.record_failure(
                provider=(
                    error.provider
                    or provider_name
                ),
                model=(
                    error.model
                    or model_name
                ),
                usage=error.usage,
                elapsed=error.metadata.get(
                    "elapsed",
                    0.0,
                ),
                error_type=error.error_type.value,
            )
            
            self.quota_store.record(
                provider=(
                    error.provider
                    or provider_name
                ),
                model=(
                    error.model
                    or model_name
                ),
                usage=error.usage,
            )

            raise

        except Exception as exc:

            error = RouterError(
                ErrorType.PROVIDER_FAILURE,
                str(exc) or "Unexpected provider failure.",
                provider=provider_name,
                model=model_name,
                retryable=True,
            )

            self.health_store.record_failure(
                error
            )

            self.metrics_store.record_failure(
                provider=provider_name,
                model=model_name,
                usage=error.usage,
                elapsed=error.metadata.get(
                    "elapsed",
                    0.0,
                ),
                error_type=error.error_type.value,
            )
            raise error from exc

        # ----------------------------------------------------
        # NORMALIZED RESULT VALIDATION
        # ----------------------------------------------------

        if not isinstance(
            result,
            ProviderResult,
        ):

            error = RouterError(
                ErrorType.PROVIDER_FAILURE,
                (
                    "Provider returned an invalid result "
                    f"type: {type(result).__name__}"
                ),
                provider=provider_name,
                model=model_name,
                retryable=True,
            )

            self.health_store.record_failure(
                error
            )
            
            self.metrics_store.record_failure(
                provider=provider_name,
                model=model_name,
                usage=error.usage,
                elapsed=error.metadata.get(
                    "elapsed",
                    0.0,
                ),
                error_type=error.error_type.value,
            )

            raise error

        if not result.success:

            error = RouterError(
                ErrorType.PROVIDER_FAILURE,
                (
                    result.error
                    or "Provider returned an unsuccessful result."
                ),
                provider=provider_name,
                model=model_name,
                status_code=result.http_status,
                retryable=True,
                metadata={
                    "provider_status": result.status,
                },
                usage=result.usage,
            )

            self.health_store.record_failure(
                error
            )
            
            self.metrics_store.record_failure(
                provider=provider_name,
                model=model_name,
                usage=error.usage,
                elapsed=error.metadata.get(
                    "elapsed",
                    0.0,
                ),
                error_type=error.error_type.value,
            )
            
            raise error

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if not result.usable:

            error = RouterError(
                ErrorType.EMPTY_RESPONSE,
                "Provider returned an unusable response.",
                provider=provider_name,
                model=model_name,
                status_code=result.http_status,
                retryable=True,
                metadata={
                    "elapsed": result.elapsed,
                },
                usage=result.usage,
            )

            self.health_store.record_failure(
                error
            )
            
            self.metrics_store.record_failure(
                provider=provider_name,
                model=model_name,
                usage=error.usage,
                elapsed=error.metadata.get(
                    "elapsed",
                    result.elapsed,
                ),
                error_type=error.error_type.value,
            )

            raise error

        self.health_store.record_success(
            provider=provider_name,
            model=model_name,
        )

        self.quota_store.record(
            provider=provider_name,
            model=model_name,
            usage=result.usage,
        )
        
        self.metrics_store.record_success(
            provider=provider_name,
            model=model_name,
            usage=result.usage,
            elapsed=result.elapsed,
        )

        return result
    
    # ========================================================
    # FINAL ERROR CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_exhausted_attempts(
        attempts: list[dict[str, Any]],
    ) -> ErrorType:
        """
        Determine the most specific error type when every
        routing candidate has failed.

        A homogeneous failure set preserves its original
        error type. Mixed failures are normalized to the
        generic provider failure category.
        """

        if not attempts:
            return ErrorType.PROVIDER_FAILURE

        error_types = {
            attempt.get("error_type")
            for attempt in attempts
            if attempt.get("error_type")
        }

        if len(error_types) != 1:
            return ErrorType.PROVIDER_FAILURE

        error_type = next(iter(error_types))

        try:
            return ErrorType(error_type)
        except ValueError:
            return ErrorType.PROVIDER_FAILURE
    
    # ========================================================
    # EXECUTE ROUTING DECISION
    # ========================================================

    def execute(
        self,
        request: ProviderRequest,
        decision: RoutingDecision,
        *,
        explicit_model: bool = False,
    ) -> ProviderResult:
        """
        Execute a previously generated routing decision.

        Candidates are attempted in ranked order.

        Retryable failures trigger fallback to the next
        candidate. Non-retryable failures stop execution.
        """
        
        if not decision.ranked:
            raise ExecutionError(
                ErrorType.MODEL_UNAVAILABLE,
                "No compatible routing candidates available.",
                retryable=False,
                metadata={
                    "candidates_considered":
                        decision.candidates_considered,
                    "candidates_available":
                        decision.candidates_available,
                    "candidates_compatible":
                        decision.candidates_compatible,
                    "routing_rejected":
                        list(decision.rejected),
                    "rejected":
                        list(decision.rejected),
                    "attempts": [],
                }
            )

        attempts: list[dict[str, Any]] = []
        
        aggregate_prompt_tokens = 0
        aggregate_completion_tokens = 0
        aggregate_total_tokens = 0

        for scored in decision.ranked:
            candidate = scored.candidate
            request_id = request.metadata.get(
                "request_id"
            )

            attempt = {
                "candidate": candidate.identifier,
                "provider": candidate.provider,
                "model": candidate.model,
                "score": scored.score,
                "request_id": request_id,
            }

            try:
                result = self.execute_candidate(
                    candidate,
                    request,
                )
                
                # --------------------------------------------------------
                # ACCOUNT SUCCESSFUL ATTEMPT USAGE
                # --------------------------------------------------------

                aggregate_prompt_tokens += (
                    result.usage.prompt_tokens
                )
                aggregate_completion_tokens += (
                    result.usage.completion_tokens
                )
                aggregate_total_tokens += (
                    result.usage.total_tokens
                )
                
                attempt["success"] = True
                attempts.append(
                    attempt
                )
                return result

            except RouterError as error:
                attempt.update(
                    {
                        "success": False,
                        "error_type":
                            error.error_type.value,
                        "error":
                            error.message,
                        "retryable":
                            error.retryable,
                        "status_code":
                            error.status_code,
                    }
                )
                aggregate_prompt_tokens += (
                    error.usage.prompt_tokens
                )
                aggregate_completion_tokens += (
                    error.usage.completion_tokens
                )
                aggregate_total_tokens += (
                    error.usage.total_tokens
                )
                attempts.append(attempt)

                # --------------------------------------------------------
                # FALLBACK POLICY
                # --------------------------------------------------------

                retryable = error.retryable

                # During automatic routing, failures that make a specific
                # candidate unusable should allow the router to continue to
                # the next ranked candidate.
                #
                # Explicit model requests remain terminal.

                if not explicit_model:

                    if error.error_type in {
                        ErrorType.MODEL_NOT_FOUND,
                        ErrorType.MODEL_UNAVAILABLE,
                        ErrorType.PAYMENT_REQUIRED,
                        ErrorType.QUOTA_EXCEEDED,
                        ErrorType.TOOL_UNSUPPORTED,
                    }:
                        retryable = True

                if not retryable:

                    raise ExecutionError(
                        error.error_type,
                        error.message,
                        provider=error.provider,
                        model=error.model,
                        status_code=error.status_code,
                        retryable=False,
                        usage=error.usage,
                        metadata={
                            "request_id":
                                request.metadata.get(
                                    "request_id"
                                ),

                            "attempts":
                                attempts,

                            "routing_rejected":
                                list(decision.rejected),
                        },
                    ) from error

                continue


        # ----------------------------------------------------
        # ALL CANDIDATES FAILED
        # ----------------------------------------------------

        last_attempt = (
            attempts[-1]
            if attempts
            else {}
        )

        final_error_type = self._classify_exhausted_attempts(
            attempts
        )

        aggregate_usage = ProviderUsage(
            prompt_tokens=aggregate_prompt_tokens,
            completion_tokens=aggregate_completion_tokens,
            total_tokens=aggregate_total_tokens,
        ).normalized()

        raise ExecutionError(
            final_error_type,
            "All routing candidates failed.",
            provider=last_attempt.get(
                "provider"
            ),
            model=last_attempt.get(
                "model"
            ),
            retryable=False,
            metadata={
                "request_id": request.metadata.get(
                    "request_id"
                ),

                "attempts": attempts,

                "routing_rejected":
                    list(decision.rejected),

                "candidates_considered":
                    decision.candidates_considered,

                "candidates_available":
                    decision.candidates_available,

                "candidates_compatible":
                    decision.candidates_compatible,

                "usage": aggregate_usage,
            },
        )

    # ========================================================
    # ROUTE + EXECUTE
    # ========================================================

    def route_and_execute(
        self,
        request: ProviderRequest,
        *,
        required_capabilities: Optional[
            set[str]
        ] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> ProviderResult:
        """
        Convenience method that performs routing and execution
        in one operation.

        The routing request receives the complete incoming request
        payload so the engine can perform request-size filtering,
        capability filtering, quota filtering, and scoring before
        any provider is executed.
        """

        # --------------------------------------------------------
        # REQUEST ID
        # --------------------------------------------------------

        request.metadata.setdefault(
            "request_id",
            create_request_id(),
        )

        # --------------------------------------------------------
        # MERGE ROUTING METADATA
        # --------------------------------------------------------

        routing_metadata = dict(
            request.metadata
        )

        if metadata:
            routing_metadata.update(
                metadata
            )

        # --------------------------------------------------------
        # ROUTING REQUEST
        # --------------------------------------------------------

        decision = self.router.route(
            RoutingRequest(
                required_capabilities=frozenset(
                    required_capabilities or []
                ),
                provider=provider,
                model=model,
                metadata=routing_metadata,

                # Pass the actual request payload into the routing
                # engine so request-size estimation can be performed.
                messages=tuple(
                    request.messages
                ),
                tools=request.tools,
                tool_choice=request.tool_choice,
            )
        )

        # --------------------------------------------------------
        # EXECUTION
        # --------------------------------------------------------

        return self.execute(
            request,
            decision,
            explicit_model=model is not None,
        )

