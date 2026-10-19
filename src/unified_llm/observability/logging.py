"""Structured logging configuration using structlog."""

import logging
import sys
from typing import Literal
import structlog
from structlog.types import Processor

from unified_llm.observability.redaction import redaction_processor


def configure_logging(
    log_level: str = "INFO",
    log_format: Literal["json", "console"] = "json",
) -> None:
    """Configure structlog and standard library logging handlers."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Base processors common to both formats
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redaction_processor,
    ]

    if log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Quiet overly chatty libraries
    for quiet_logger in ("uvicorn.access", "httpcore", "httpx"):
        logging.getLogger(quiet_logger).setLevel(logging.WARNING)
