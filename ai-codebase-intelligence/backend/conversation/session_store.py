"""Redis-backed conversation session store."""

from __future__ import annotations

import json
import time
import uuid

import redis

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

SESSION_KEY_PREFIX = "conversation:session:"
SESSION_TTL_SECONDS = 60 * 60 * 24
MAX_CONTENT_CHARS = 8000
_ALLOWED_ROLES = {"user", "assistant"}


def _get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


class SessionStore:
    """Store chat turns in Redis as JSON arrays per session."""

    def __init__(self) -> None:
        self._redis = _get_redis_client()

    def create_session(self) -> str:
        return str(uuid.uuid4())

    def session_exists(self, session_id: str) -> bool:
        key = self._key(session_id)
        try:
            return bool(self._redis.exists(key))
        except redis.RedisError as exc:
            logger.warning("session_store_exists_failed", session_id=session_id, error=str(exc))
            return False

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        provider_used: str | None = None,
        model_used: str | None = None,
    ) -> bool:
        if role not in _ALLOWED_ROLES:
            return False

        clean = (content or "").strip()
        if len(clean) > MAX_CONTENT_CHARS:
            clean = clean[: MAX_CONTENT_CHARS - 12].rstrip() + "[truncated]"

        turn = {
            "role": role,
            "content": clean,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if sources:
            turn["sources"] = sources
        if provider_used:
            turn["provider_used"] = provider_used
        if model_used:
            turn["model_used"] = model_used

        key = self._key(session_id)
        try:
            turns = self.get_turns(session_id)
            turns.append(turn)
            self._redis.setex(key, SESSION_TTL_SECONDS, json.dumps(turns))
            return True
        except redis.RedisError as exc:
            logger.warning("session_store_append_failed", session_id=session_id, error=str(exc))
            return False

    def get_turns(self, session_id: str) -> list[dict]:
        key = self._key(session_id)
        try:
            raw = self._redis.get(key)
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, redis.RedisError) as exc:
            logger.warning("session_store_get_failed", session_id=session_id, error=str(exc))
            return []

    def turn_count(self, session_id: str) -> int:
        return len(self.get_turns(session_id))

    def delete_session(self, session_id: str) -> bool:
        key = self._key(session_id)
        try:
            return bool(self._redis.delete(key))
        except redis.RedisError as exc:
            logger.warning("session_store_delete_failed", session_id=session_id, error=str(exc))
            return False

    @staticmethod
    def _key(session_id: str) -> str:
        return SESSION_KEY_PREFIX + session_id
