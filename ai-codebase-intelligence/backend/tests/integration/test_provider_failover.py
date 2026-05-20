"""Integration-style tests for provider failover orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import LLMProviderError
from reasoning.llm_router import TaskType, ask


@pytest.fixture
def all_keys_set():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = "key"
        mock_settings.openrouter_api_key = "key"
        yield mock_settings


@pytest.mark.asyncio
async def test_failover_uses_next_provider_after_primary_failure(all_keys_set):
    primary = AsyncMock()
    primary.complete = AsyncMock(side_effect=LLMProviderError("primary unavailable"))

    fallback = AsyncMock()
    fallback.complete = AsyncMock(return_value="fallback answer")

    call_count = 0

    def _factory(provider, model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return primary
        return fallback

    circuit = MagicMock()
    circuit.is_available.return_value = True

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        with patch("reasoning.llm_router._get_client", side_effect=_factory):
            answer_text, provider, model = await ask(TaskType.CODE_QA, "question")

    assert answer_text == "fallback answer"
    assert call_count >= 2
    assert circuit.record_failure.call_count >= 1
    assert circuit.record_success.call_count >= 1


@pytest.mark.asyncio
async def test_raises_when_all_providers_fail(all_keys_set):
    failing = AsyncMock()
    failing.complete = AsyncMock(side_effect=LLMProviderError("all down"))

    circuit = MagicMock()
    circuit.is_available.return_value = True

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        with patch("reasoning.llm_router._get_client", return_value=failing):
            with pytest.raises(LLMProviderError, match="All LLM providers failed"):
                await ask(TaskType.REASONING, "question")


@pytest.mark.asyncio
async def test_open_circuit_provider_is_skipped(all_keys_set):
    client = AsyncMock()
    client.complete = AsyncMock(return_value="ok")

    def _available(provider_name: str) -> bool:
        # DeepSeek first for CODE_QA; force it open to validate skip.
        return provider_name != "deepseek"

    circuit = MagicMock()
    circuit.is_available.side_effect = _available

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        with patch("reasoning.llm_router._get_client", return_value=client) as client_factory:
            answer_text, provider, model = await ask(TaskType.CODE_QA, "question")

    assert answer_text == "ok"
    # Skipped deepseek, then moved to next available provider.
    assert provider.value in {"gemini", "openrouter"}
    assert client_factory.call_count >= 1


@pytest.mark.asyncio
async def test_provider_failure_records_circuit_failure(all_keys_set):
    client = AsyncMock()
    client.complete = AsyncMock(side_effect=RuntimeError("503 Service Unavailable"))

    circuit = MagicMock()
    circuit.is_available.return_value = True

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        with patch("reasoning.llm_router._get_client", return_value=client):
            with pytest.raises(LLMProviderError):
                await ask(TaskType.SUMMARIZE, "question")

    assert circuit.record_failure.call_count >= 1


@pytest.mark.asyncio
async def test_success_after_half_open_records_success(all_keys_set):
    client = AsyncMock()
    client.complete = AsyncMock(return_value="healthy again")

    circuit = MagicMock()
    circuit.is_available.return_value = True

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        with patch("reasoning.llm_router._get_client", return_value=client):
            answer_text, provider, model = await ask(TaskType.ARCHITECTURE, "question")

    assert answer_text == "healthy again"
    assert circuit.record_success.call_count >= 1
