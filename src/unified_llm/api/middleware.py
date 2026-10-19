"""API Middlewares: Request ID, Inbound Authentication, and Payload Size Limit."""

import secrets
import uuid
from typing import Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import structlog

from unified_llm.config.settings import ServiceSettings

# Paths that do not require authentication
UNAUTHENTICATED_PATHS = {
    "/healthz",
    "/readyz",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns or propagates X-Request-ID and binds it to structlog contextvars."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming_id = request.headers.get("X-Request-ID")
        req_id = incoming_id if incoming_id else f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = req_id

        # Bind to structlog
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=req_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response


class InboundAuthMiddleware(BaseHTTPMiddleware):
    """Validates inbound Bearer token against ULIL_API_KEYS using constant-time comparison."""

    def __init__(self, app: Any, settings: ServiceSettings) -> None:
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # If no API keys are configured, authentication is disabled
        if not self.settings.api_keys or request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        req_id = getattr(request.state, "request_id", None)

        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "unauthorized",
                        "message": "Missing or malformed Authorization header. Expected 'Bearer <key>'.",
                        "provider": None,
                        "retryable": False,
                        "retry_after_s": None,
                        "provider_status": 401,
                        "request_id": req_id,
                        "details": {},
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1].strip()
        is_valid = any(
            secrets.compare_digest(token, secret_key.get_secret_value())
            for secret_key in self.settings.api_keys
        )

        if not is_valid:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "type": "unauthorized",
                        "message": "Invalid API key.",
                        "provider": None,
                        "retryable": False,
                        "retry_after_s": None,
                        "provider_status": 401,
                        "request_id": req_id,
                        "details": {},
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)


class PayloadSizeLimitMiddleware(BaseHTTPMiddleware):
    """Enforces maximum inbound request body size."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        req_id = getattr(request.state, "request_id", None)

        if content_length:
            try:
                length_int = int(content_length)
                if length_int > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "type": "payload_too_large",
                                "message": f"Payload size ({length_int} bytes) exceeds limit ({self.max_bytes} bytes).",
                                "provider": None,
                                "retryable": False,
                                "retry_after_s": None,
                                "provider_status": 413,
                                "request_id": req_id,
                                "details": {"max_bytes": self.max_bytes},
                            }
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)
