import csv
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from backend.app.core.config import LATENCY_LOG_FILE, API_LOG_FILE, LOGS_DIR

# Ensure log directory exists
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

# Write CSV header if file doesn't exist
if not Path(LATENCY_LOG_FILE).exists():
    with open(LATENCY_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "method", "path", "query_params",
            "status_code", "latency_ms", "cached"
        ])


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Records the wall-clock latency for every request
    2. Writes it to logs/latency_log.csv
    3. Logs it to console with loguru
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Record start time
        t_start = time.time()

        # Execute the actual request handler
        response = await call_next(request)

        # Compute latency
        latency_ms = (time.time() - t_start) * 1000

        # Extract request metadata
        method      = request.method
        path        = request.url.path
        query_str   = str(request.query_params)
        status_code = response.status_code
        timestamp   = datetime.now(timezone.utc).isoformat()

        # Determine if it was a cache hit from response headers
        cached = response.headers.get("X-Cache", "miss") == "hit"

        # Log to console
        level = "INFO" if status_code < 400 else "WARNING"
        getattr(logger, level.lower())(
            f"  {method} {path}?{query_str} "
            f"→ {status_code} "
            f"({latency_ms:.1f}ms) "
            f"{'[CACHED]' if cached else ''}"
        )

        # Write to CSV
        try:
            with open(LATENCY_LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, method, path, query_str,
                    status_code, round(latency_ms, 2), cached
                ])
        except Exception as e:
            logger.warning(f"Failed to write latency log: {e}")

        # Write structured event log (JSON Lines)
        try:
            event = {
                "event_type": "request",
                "timestamp":  timestamp,
                "method":     method,
                "path":       path,
                "params":     query_str,
                "status":     status_code,
                "latency_ms": round(latency_ms, 2),
                "cached":     cached,
            }
            with open(API_LOG_FILE, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSON log: {e}")

        # Add latency header (useful for debugging in Postman)
        response.headers["X-Latency-Ms"]    = str(round(latency_ms, 2))
        response.headers["X-Cache"]         = "hit" if cached else "miss"

        return response