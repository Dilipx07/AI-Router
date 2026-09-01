from router_api import (
    ERROR_HTTP_STATUS,
    error_http_status,
)

from router.errors import ErrorType
from router.executor import ExecutionError


print("=== API ERROR MAPPING TEST ===")


expected = {
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


for error_type, status in expected.items():

    error = ExecutionError(
        error_type,
        "test error",
    )

    actual = error_http_status(
        error
    )

    print(
        f"{error_type.value:22} "
        f"→ HTTP {actual}"
    )

    assert actual == status


print()
print("all error mappings: PASS")
print("RESULT: PASS")
