import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from contextvars import ContextVar

from app.metrics import (
    RECOMMENDATION_LATENCY,
    RECOMMENDATION_REQUESTS,
    CACHE_HIT_RATIO,
)

# Context variable to pass correlation ID through async call stack
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="unknown")

logger = structlog.get_logger()

# Rolling cache hit tracker (simple, no external state)
_total_requests: int = 0
_cache_hits: int = 0


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Records per-request Prometheus metrics.

    Metrics recorded:
      - RECOMMENDATION_LATENCY (only for /recommend)
      - RECOMMENDATION_REQUESTS (only for /recommend)
      - CACHE_HIT_RATIO (rolling, updated on every /recommend call)

    Why apply only to /recommend?
      Generic HTTP metrics (all endpoints) are already captured by
      prometheus_fastapi_instrumentator. This middleware adds
      BUSINESS metrics that are specific to our recommendation logic.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        # Set correlation ID for this request
        request_id = str(uuid.uuid4())[:8]
        REQUEST_ID.set(request_id)
        request.state.request_id = request_id

        response = await call_next(request)

        elapsed = time.perf_counter() - start

        # Only track recommendation-specific metrics on /recommend
        if request.url.path == "/recommend":
            global _total_requests, _cache_hits

            variant = response.headers.get("X-AB-Variant", "unknown")
            cache_status = response.headers.get("X-Cache-Status", "miss")
            status = "cache_hit" if cache_status == "hit" else (
                "success" if response.status_code == 200 else "error"
            )

            RECOMMENDATION_LATENCY.labels(variant=variant).observe(elapsed)
            RECOMMENDATION_REQUESTS.labels(
                variant=variant, status=status
            ).inc()

            # Update rolling cache ratio
            _total_requests += 1
            if cache_status == "hit":
                _cache_hits += 1
            CACHE_HIT_RATIO.set(
                _cache_hits / _total_requests if _total_requests > 0 else 0.0
            )

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Emits one structured JSON log line per request.

    Example output (pretty-printed for readability):
    {
      "event": "http_request",
      "request_id": "a3f8b1c2",
      "method": "GET",
      "path": "/recommend",
      "status_code": 200,
      "duration_ms": 18.4,
      "variant": "sasrec",
      "cache_status": "miss",
      "timestamp": "2025-01-15T10:30:00.123Z"
    }
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        log_data = {
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed_ms, 2),
        }

        # Add recommendation-specific fields when available
        if request.url.path == "/recommend":
            log_data["variant"] = response.headers.get("X-AB-Variant", "unknown")
            log_data["cache_status"] = response.headers.get("X-Cache-Status", "miss")

        if response.status_code >= 500:
            logger.error("http_request", **log_data)
        elif response.status_code >= 400:
            logger.warning("http_request", **log_data)
        else:
            logger.info("http_request", **log_data)

        return response