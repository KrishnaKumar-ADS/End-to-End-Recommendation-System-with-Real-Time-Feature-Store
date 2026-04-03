from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    # Default limit applies to ANY endpoint decorated with @limiter.limit
    # but no limit is set globally — each endpoint opts in explicitly.
    default_limits=[],
    # Store rate limit state in Redis (survives app restarts)
    # If Redis is down, rate limiting degrades gracefully (no crash).
    storage_uri="redis://localhost:6379/1",  # DB 1 (separate from rec cache)
)