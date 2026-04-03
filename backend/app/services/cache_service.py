import json
import time
from typing import Optional, Any
from loguru import logger

try:
    import redis as redis_lib
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis package not installed. Caching disabled.")

from backend.app.core.config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    REDIS_CACHE_TTL, REDIS_ENABLED
)


class CacheService:
    """
    Redis-backed result cache for recommendation responses.
    Gracefully degrades to no-op if Redis is unavailable.
    """

    def __init__(self):
        self.requested_enabled = REDIS_ENABLED and REDIS_AVAILABLE
        self.enabled = self.requested_enabled
        self.client  = None
        self.connection_error: Optional[str] = None
        if not REDIS_ENABLED:
            self.disabled_reason = "REDIS_ENABLED=false"
        elif not REDIS_AVAILABLE:
            self.disabled_reason = "redis package not installed"
        else:
            self.disabled_reason = ""
        self._connect()

    def _connect(self):
        """Attempt to connect to Redis. Disable caching if it fails."""
        if not self.enabled:
            logger.info(f"Cache: Disabled ({self.disabled_reason})")
            return
        try:
            self.client = redis_lib.Redis(
                host               = REDIS_HOST,
                port               = REDIS_PORT,
                db                 = REDIS_DB,
                socket_connect_timeout = 2,
                decode_responses   = True,   # Return str, not bytes
            )
            self.client.ping()
            self.connection_error = None
            logger.info(f"Cache: ✅ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Cache: ⚠️ Redis unavailable ({e}). Caching disabled.")
            self.client  = None
            self.enabled = False
            self.connection_error = str(e)
            self.disabled_reason = "connection failed"

    def _make_key(self, user_id: int, top_k: int) -> str:
        """Generate a consistent Redis key for a user's recommendations."""
        return f"rec:{user_id}:{top_k}"

    def get(self, user_id: int, top_k: int) -> Optional[Any]:
        """
        Retrieve cached recommendations.
        Returns: parsed dict if cache hit, None if miss or disabled.
        """
        if not self.enabled or self.client is None:
            return None
        try:
            key  = self._make_key(user_id, top_k)
            data = self.client.get(key)
            if data is not None:
                logger.debug(f"Cache HIT: user_id={user_id}, top_k={top_k}")
                return json.loads(data)
            logger.debug(f"Cache MISS: user_id={user_id}, top_k={top_k}")
            return None
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
            return None

    def set(self, user_id: int, top_k: int, value: Any, ttl: int = REDIS_CACHE_TTL):
        """
        Store recommendations in cache.
        value: any JSON-serializable object (list of dicts)
        ttl:   seconds until expiry (default 5 minutes)
        """
        if not self.enabled or self.client is None:
            return
        try:
            key  = self._make_key(user_id, top_k)
            data = json.dumps(value)
            self.client.setex(key, ttl, data)
            logger.debug(f"Cache SET: user_id={user_id}, top_k={top_k}, ttl={ttl}s")
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")

    def invalidate(self, user_id: int, top_k: Optional[int] = None):
        """
        Invalidate cache for a user.
        Called after receiving user feedback — stale results should be refreshed.
        If top_k=None, invalidates all top_k variants for this user.
        """
        if not self.enabled or self.client is None:
            return
        try:
            if top_k is not None:
                key = self._make_key(user_id, top_k)
                self.client.delete(key)
            else:
                # Delete all keys matching rec:{user_id}:*
                pattern = f"rec:{user_id}:*"
                keys    = self.client.keys(pattern)
                if keys:
                    self.client.delete(*keys)
                    logger.debug(f"Cache INVALIDATE: {len(keys)} keys for user_id={user_id}")
        except Exception as e:
            logger.warning(f"Cache INVALIDATE error: {e}")

    def health_check(self) -> dict:
        """Returns cache status for /health endpoint."""
        if not self.requested_enabled:
            return {"status": "disabled", "detail": self.disabled_reason}
        if self.client is None:
            return {
                "status": "unavailable",
                "detail": self.connection_error or "redis client unavailable",
            }
        try:
            self.client.ping()
            info = self.client.info("memory")
            used_mb = info.get("used_memory", 0) / 1024 / 1024
            return {
                "status": "healthy",
                "detail": f"{REDIS_HOST}:{REDIS_PORT}, {used_mb:.1f} MB used"
            }
        except Exception as e:
            return {"status": "unavailable", "detail": str(e)}