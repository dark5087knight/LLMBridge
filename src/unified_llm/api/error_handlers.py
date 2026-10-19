"""FastAPI exception handlers mapping all errors to the standardized error envelope."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from structlog import get_logger

from unified_llm.core.errors import ULILError

logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Register all custom error handlers on the FastAPI application."""

    @app.exception_handler(ULILError)
    async def ulil_error_handler(request: Request, exc: ULILError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "ulil_error_handled",
            error_type=exc.type,
            status_code=exc.http_status,
            provider=exc.provider,
            request_id=request_id,
            error_message=exc.message,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_envelope(request_id=request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "request_validation_failed",
            request_id=request_id,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "invalid_request",
                    "message": "The request payload failed schema validation.",
                    "provider": None,
                    "retryable": False,
                    "retry_after_s": None,
                    "provider_status": 422,
                    "request_id": request_id,
                    "details": {"validation_errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "message": exc.detail,
                    "provider": None,
                    "retryable": False,
                    "retry_after_s": None,
                    "provider_status": exc.status_code,
                    "request_id": request_id,
                    "details": {},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "unhandled_server_exception",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "message": "An unexpected internal server error occurred.",
                    "provider": None,
                    "retryable": False,
                    "retry_after_s": None,
                    "provider_status": 500,
                    "request_id": request_id,
                    "details": {},
                }
            },
        )
