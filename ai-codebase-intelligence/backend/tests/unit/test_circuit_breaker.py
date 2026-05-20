"""Unit tests for reasoning/circuit_breaker.py."""

import time
from unittest.mock import patch

import pytest

from reasoning.circuit_breaker import (
    FAILURE_THRESHOLD,
    RECOVERY_TIMEOUT_SEC,
    CircuitBreaker,
    CircuitState,
)


@pytest.fixture
def cb():
    import fakeredis

    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("reasoning.circuit_breaker.redis.Redis.from_url", return_value=fake):
        yield CircuitBreaker()


def test_new_provider_is_available(cb):
    assert cb.is_available("gemini") is True


def test_single_failure_does_not_open_circuit(cb):
    cb.record_failure("gemini")
    assert cb.is_available("gemini") is True


def test_threshold_failures_open_circuit(cb):
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("gemini")
    assert cb.is_available("gemini") is False


def test_success_resets_failure_count(cb):
    cb.record_failure("gemini")
    cb.record_failure("gemini")
    cb.record_success("gemini")

    assert cb.is_available("gemini") is True
    status = cb.get_all_statuses()["gemini"]
    assert status.failure_count == 0


def test_circuit_reopens_after_success_then_3_more_failures(cb):
    cb.record_success("gemini")
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("gemini")
    assert cb.is_available("gemini") is False


def test_open_circuit_recovers_after_timeout(cb):
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("gemini")
    assert cb.is_available("gemini") is False

    future_time = time.time() + RECOVERY_TIMEOUT_SEC + 1
    with patch("reasoning.circuit_breaker.time.time", return_value=future_time):
        assert cb.is_available("gemini") is True


def test_providers_are_independent(cb):
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("gemini")

    assert cb.is_available("deepseek") is True
    assert cb.is_available("openrouter") is True
    assert cb.is_available("gemini") is False


def test_reset_closes_open_circuit(cb):
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("openrouter")
    assert cb.is_available("openrouter") is False

    cb.reset("openrouter")
    assert cb.is_available("openrouter") is True


def test_get_all_statuses_returns_all_providers(cb):
    statuses = cb.get_all_statuses()
    assert "gemini" in statuses
    assert "deepseek" in statuses
    assert "openrouter" in statuses


def test_status_state_is_closed_by_default(cb):
    status = cb.get_all_statuses()["gemini"]
    assert status.state == CircuitState.CLOSED


def test_status_state_is_open_after_threshold(cb):
    for _ in range(FAILURE_THRESHOLD):
        cb.record_failure("deepseek")
    status = cb.get_all_statuses()["deepseek"]
    assert status.state == CircuitState.OPEN
