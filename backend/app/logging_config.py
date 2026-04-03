import logging
import os
import sys
import structlog
from pathlib import Path


def configure_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Configure structlog for the application.
    Call this ONCE at startup in main.py before any other code runs.

    Args:
        log_level: "DEBUG" | "INFO" | "WARNING" | "ERROR"
        log_dir:   Directory for log files (created if not exists)
    """

    env = os.getenv("ENV", "dev")
    Path(log_dir).mkdir(exist_ok=True)

    # ── Configure standard library logging (structlog sits on top) ──
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    # ── Shared processors (run for every log event) ──────────────────
    shared_processors = [
        structlog.contextvars.merge_contextvars,  # thread-local context
        structlog.stdlib.add_log_level,           # add "level" key
        structlog.stdlib.add_logger_name,         # add "logger" key
        structlog.processors.TimeStamper(fmt="iso", utc=True),  # ISO timestamp
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,     # pretty exception formatting
    ]

    # ── Production: JSON output ──────────────────────────────────────
    if env == "prod":
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colored, human-readable output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

