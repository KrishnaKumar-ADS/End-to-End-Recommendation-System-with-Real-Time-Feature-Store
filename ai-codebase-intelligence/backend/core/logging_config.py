import logging
import os
import sys
from typing import Any

import structlog


_CONFIGURED = False


def configure_logging() -> None:
    """Configure structured logging for development and production."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    app_env = os.getenv("APP_ENV", "development").lower()
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        structlog.processors.StackInfoRenderer(),
    ]

    if app_env == "production":
        renderer = structlog.processors.JSONRenderer()
        stream = sys.stdout
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
        stream = sys.stderr

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", level=log_level, stream=stream, force=True)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> structlog.types.FilteringBoundLogger:
    return structlog.get_logger(name)
