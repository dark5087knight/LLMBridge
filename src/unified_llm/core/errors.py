"""Error taxonomy and error envelope formatting for ULIL."""

from typing import Any


class ULILError(Exception):
    """Base exception for all ULIL errors."""

    type: str = "internal_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        provider_status: int | None = None,
        retry_after_s: float | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.provider_status = provider_status
        self.retry_after_s = retry_after_s
        self.details = details or {}
        if retryable is not None:
            self.retryable = retryable

    def to_envelope(self, request_id: str | None = None) -> dict[str, Any]:
        """Format into standardized error envelope."""
        return {
            "error": {
                "type": self.type,
                "message": self.message,
                "provider": self.provider,
                "retryable": self.retryable,
                "retry_after_s": self.retry_after_s,
                "provider_status": self.provider_status,
                "request_id": request_id,
                "details": self.details,
            }
        }


# ============================================================================
# Client Errors (4xx)
# ============================================================================


class ClientError(ULILError):
    """Base for client-side errors."""

    http_status = 400
    retryable = False


class InvalidRequest(ClientError):
    type = "invalid_request"
    http_status = 422


class UnsupportedParameter(ClientError):
    type = "unsupported_parameter"
    http_status = 422


class UnsupportedFeature(ClientError):
    type = "unsupported_feature"
    http_status = 422


class UnknownProvider(ClientError):
    type = "unknown_provider"
    http_status = 404


class ModelNotAllowed(ClientError):
    type = "model_not_allowed"
    http_status = 403


# ============================================================================
# Provider Errors (Upstream errors)
# ============================================================================


class ProviderError(ULILError):
    """Base for provider/upstream errors."""

    http_status = 502
    retryable = False


class ProviderAuthError(ProviderError):
    type = "provider_auth_error"
    http_status = 401
    retryable = False


class ProviderNotConfigured(ProviderError):
    type = "provider_not_configured"
    http_status = 503
    retryable = False


class ModelNotFound(ProviderError):
    type = "model_not_found"
    http_status = 404
    retryable = False


class ContextLengthExceeded(ProviderError):
    type = "context_length_exceeded"
    http_status = 413
    retryable = False


class ContentRejected(ProviderError):
    type = "content_rejected"
    http_status = 422
    retryable = False


class RateLimited(ProviderError):
    type = "rate_limited"
    http_status = 429
    retryable = True


class ProviderTimeout(ProviderError):
    type = "provider_timeout"
    http_status = 504
    retryable = True


class ProviderConnectionError(ProviderError):
    type = "provider_connection_error"
    http_status = 502
    retryable = True


class ProviderUnavailable(ProviderError):
    type = "provider_unavailable"
    http_status = 503
    retryable = True


class BadProviderResponse(ProviderError):
    type = "bad_provider_response"
    http_status = 502
    retryable = True


# ============================================================================
# Internal Errors (500)
# ============================================================================


class InternalError(ULILError):
    type = "internal_error"
    http_status = 500
    retryable = False
