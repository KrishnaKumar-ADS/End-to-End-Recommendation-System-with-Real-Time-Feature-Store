"""Redis-backed circuit breaker for LLM providers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum

import redis

from core.config import get_settings
from core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

FAILURE_THRESHOLD = 3
RECOVERY_TIMEOUT_SEC = 60
STATE_TTL_SEC = 60 * 60 * 24
CIRCUIT_KEY_PREFIX = "llm:circuit:"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStatus:
    provider: str
    state: CircuitState
    failure_count: int
    last_failure_time: float
    last_success_time: float


class CircuitBreaker:
    """Circuit breaker with one independent state per provider."""

    def __init__(self) -> None:
        self._redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    def is_available(self, provider: str) -> bool:
        try:
            status = self._load_status(provider)

            if status.state == CircuitState.CLOSED:
                return True

            if status.state == CircuitState.OPEN:
                elapsed = time.time() - status.last_failure_time
                if elapsed >= RECOVERY_TIMEOUT_SEC:
                    self._set_state(provider, CircuitState.HALF_OPEN)
                    logger.info(
                        "circuit_half_open",
                        provider=provider,
                        elapsed_secs=round(elapsed, 1),
                    )
                    return True

                logger.debug(
                    "circuit_open_skip",
                    provider=provider,
                    remaining_cooldown_s=round(RECOVERY_TIMEOUT_SEC - elapsed, 1),
                )
                return False

            if status.state == CircuitState.HALF_OPEN:
                return True

            return True
        except redis.RedisError as exc:
            logger.warning("circuit_check_failed", provider=provider, error=str(exc))
            return True

    def record_success(self, provider: str) -> None:
        try:
            status = self._load_status(provider)
            if status.state != CircuitState.CLOSED or status.failure_count > 0:
                self._save_status(
                    provider,
                    CircuitStatus(
                        provider=provider,
                        state=CircuitState.CLOSED,
                        failure_count=0,
                        last_failure_time=status.last_failure_time,
                        last_success_time=time.time(),
                    ),
                )
                logger.info("circuit_closed", provider=provider)
        except redis.RedisError as exc:
            logger.warning("circuit_record_success_failed", provider=provider, error=str(exc))

    def record_failure(self, provider: str) -> None:
        try:
            status = self._load_status(provider)
            new_count = status.failure_count + 1
            now = time.time()
            if new_count >= FAILURE_THRESHOLD:
                new_state = CircuitState.OPEN
                logger.warning(
                    "circuit_opened",
                    provider=provider,
                    failure_count=new_count,
                    cooldown_secs=RECOVERY_TIMEOUT_SEC,
                )
            else:
                new_state = CircuitState.CLOSED
                logger.info(
                    "circuit_failure_recorded",
                    provider=provider,
                    failure_count=new_count,
                    threshold=FAILURE_THRESHOLD,
                )

            self._save_status(
                provider,
                CircuitStatus(
                    provider=provider,
                    state=new_state,
                    failure_count=new_count,
                    last_failure_time=now,
                    last_success_time=status.last_success_time,
                ),
            )
        except redis.RedisError as exc:
            logger.warning("circuit_record_failure_failed", provider=provider, error=str(exc))

    def get_all_statuses(self) -> dict[str, CircuitStatus]:
        providers = ["gemini", "deepseek", "openrouter"]
        return {provider: self._load_status(provider) for provider in providers}

    def reset(self, provider: str) -> None:
        try:
            self._redis.delete(CIRCUIT_KEY_PREFIX + provider)
            logger.info("circuit_manually_reset", provider=provider)
        except redis.RedisError as exc:
            logger.warning("circuit_reset_failed", provider=provider, error=str(exc))

    def _load_status(self, provider: str) -> CircuitStatus:
        key = CIRCUIT_KEY_PREFIX + provider
        try:
            raw = self._redis.get(key)
            if raw:
                data = json.loads(raw)
                return CircuitStatus(
                    provider=provider,
                    state=CircuitState(data.get("state", CircuitState.CLOSED.value)),
                    failure_count=int(data.get("failure_count", 0)),
                    last_failure_time=float(data.get("last_failure_time", 0.0)),
                    last_success_time=float(data.get("last_success_time", 0.0)),
                )
        except (json.JSONDecodeError, redis.RedisError, ValueError):
            pass

        return CircuitStatus(
            provider=provider,
            state=CircuitState.CLOSED,
            failure_count=0,
            last_failure_time=0.0,
            last_success_time=0.0,
        )

    def _save_status(self, provider: str, status: CircuitStatus) -> None:
        key = CIRCUIT_KEY_PREFIX + provider
        payload = {
            "state": status.state.value,
            "failure_count": status.failure_count,
            "last_failure_time": status.last_failure_time,
            "last_success_time": status.last_success_time,
        }
        self._redis.setex(key, STATE_TTL_SEC, json.dumps(payload))

    def _set_state(self, provider: str, new_state: CircuitState) -> None:
        status = self._load_status(provider)
        status.state = new_state
        self._save_status(provider, status)
