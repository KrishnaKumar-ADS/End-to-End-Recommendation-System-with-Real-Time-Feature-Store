"""Unit tests for ask route short-circuit fallback heuristics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from api.routes.ask import _SHORT_CIRCUIT_FAILURE_WINDOW_SEC, _should_short_circuit_llm_attempts
from reasoning.llm_router import Provider


def _status(*, failure_count: int, last_failure_time: float, last_success_time: float = 0.0):
    return SimpleNamespace(
        failure_count=failure_count,
        last_failure_time=last_failure_time,
        last_success_time=last_success_time,
    )


def test_short_circuit_true_when_all_providers_recently_failed():
    now = 10_000.0
    statuses = {
        "gemini": _status(failure_count=1, last_failure_time=now - 10),
        "deepseek": _status(failure_count=2, last_failure_time=now - 25),
        "openrouter": _status(failure_count=1, last_failure_time=now - 5),
    }

    with patch(
        "api.routes.ask.get_available_providers",
        return_value=[Provider.GEMINI, Provider.DEEPSEEK, Provider.OPENROUTER],
    ):
        with patch("api.routes.ask.CircuitBreaker") as breaker_cls:
            breaker_cls.return_value.get_all_statuses.return_value = statuses
            with patch("api.routes.ask._time.time", return_value=now):
                assert _should_short_circuit_llm_attempts() is True


def test_short_circuit_false_when_any_provider_failure_is_stale():
    now = 10_000.0
    stale_failure = now - (_SHORT_CIRCUIT_FAILURE_WINDOW_SEC + 5)
    statuses = {
        "gemini": _status(failure_count=1, last_failure_time=now - 10),
        "deepseek": _status(failure_count=1, last_failure_time=stale_failure),
        "openrouter": _status(failure_count=1, last_failure_time=now - 3),
    }

    with patch(
        "api.routes.ask.get_available_providers",
        return_value=[Provider.GEMINI, Provider.DEEPSEEK, Provider.OPENROUTER],
    ):
        with patch("api.routes.ask.CircuitBreaker") as breaker_cls:
            breaker_cls.return_value.get_all_statuses.return_value = statuses
            with patch("api.routes.ask._time.time", return_value=now):
                assert _should_short_circuit_llm_attempts() is False


def test_short_circuit_false_when_provider_recovered_after_failure():
    now = 10_000.0
    statuses = {
        "gemini": _status(failure_count=1, last_failure_time=now - 10, last_success_time=now - 1),
        "deepseek": _status(failure_count=1, last_failure_time=now - 10),
        "openrouter": _status(failure_count=1, last_failure_time=now - 10),
    }

    with patch(
        "api.routes.ask.get_available_providers",
        return_value=[Provider.GEMINI, Provider.DEEPSEEK, Provider.OPENROUTER],
    ):
        with patch("api.routes.ask.CircuitBreaker") as breaker_cls:
            breaker_cls.return_value.get_all_statuses.return_value = statuses
            with patch("api.routes.ask._time.time", return_value=now):
                assert _should_short_circuit_llm_attempts() is False
