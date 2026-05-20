from core.logging_config import configure_logging, get_logger


def setup_logging() -> None:
    """Backward-compatible wrapper used across the codebase."""
    configure_logging()