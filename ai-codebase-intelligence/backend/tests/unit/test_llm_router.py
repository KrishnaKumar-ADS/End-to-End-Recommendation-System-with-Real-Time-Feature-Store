"""Unit tests for reasoning.llm_router full Week 6 behavior."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.exceptions import LLMProviderError
from reasoning.llm_router import (
    Provider,
    TaskType,
    _default_temperature,
    ask,
    get_available_providers,
    route,
    stream_ask,
)


@pytest.fixture(autouse=True)
def mock_circuit_breaker():
    circuit = MagicMock()
    circuit.is_available.return_value = True
    circuit.record_success.return_value = None
    circuit.record_failure.return_value = None

    with patch("reasoning.llm_router._circuit_breaker", circuit):
        yield circuit


@pytest.fixture
def all_keys_set():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = "key"
        mock_settings.openrouter_api_key = "key"
        yield mock_settings


@pytest.fixture
def only_gemini():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = "key"
        mock_settings.deepseek_api_key = ""
        mock_settings.openrouter_api_key = ""
        yield mock_settings


@pytest.fixture
def no_keys():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = ""
        mock_settings.deepseek_api_key = ""
        mock_settings.openrouter_api_key = ""
        yield mock_settings


def test_route_code_qa_prefers_deepseek(all_keys_set):
    assert route(TaskType.CODE_QA) == Provider.DEEPSEEK


def test_route_reasoning_prefers_gemini(all_keys_set):
    assert route(TaskType.REASONING) == Provider.GEMINI


def test_route_security_prefers_deepseek(all_keys_set):
    assert route(TaskType.SECURITY) == Provider.DEEPSEEK


def test_route_falls_back_to_gemini_when_deepseek_missing(only_gemini):
    assert route(TaskType.CODE_QA) == Provider.GEMINI


def test_route_raises_when_no_provider(no_keys):
    with pytest.raises(RuntimeError, match="No LLM provider available"):
        route(TaskType.CODE_QA)


def test_available_providers_reflect_keys(all_keys_set):
    available = get_available_providers()
    assert Provider.GEMINI in available
    assert Provider.DEEPSEEK in available
    assert Provider.OPENROUTER in available


def test_available_providers_with_only_gemini(only_gemini):
    available = get_available_providers()
    assert available == [Provider.GEMINI]


def test_default_temperature_for_code_is_low():
    assert _default_temperature(TaskType.CODE_QA) <= 0.2


def test_default_temperature_reasoning_above_code():
    assert _default_temperature(TaskType.REASONING) > _default_temperature(TaskType.CODE_QA)


@pytest.mark.asyncio
async def test_ask_returns_answer_and_provider(all_keys_set):
    client = AsyncMock()
    client.complete = AsyncMock(return_value="ok")

    with patch("reasoning.llm_router._get_client", return_value=client):
        answer, provider, model = await ask(TaskType.CODE_QA, "question")

    assert answer == "ok"
    assert provider in (Provider.DEEPSEEK, Provider.GEMINI, Provider.OPENROUTER)
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_ask_falls_back_on_provider_error(all_keys_set):
    primary = AsyncMock()
    primary.complete = AsyncMock(side_effect=LLMProviderError("primary failed"))

    fallback = AsyncMock()
    fallback.complete = AsyncMock(return_value="fallback")

    call_count = 0

    def factory(provider, model):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return primary
        return fallback

    with patch("reasoning.llm_router._get_client", side_effect=factory):
        answer, provider, model = await ask(TaskType.CODE_QA, "question")

    assert answer == "fallback"


@pytest.mark.asyncio
async def test_ask_raises_when_every_provider_fails(all_keys_set):
    failing = AsyncMock()
    failing.complete = AsyncMock(side_effect=LLMProviderError("failed"))

    with patch("reasoning.llm_router._get_client", return_value=failing):
        with pytest.raises(LLMProviderError, match="All LLM providers failed"):
            await ask(TaskType.CODE_QA, "question")


@pytest.mark.asyncio
async def test_ask_skips_providers_without_keys():
    with patch("reasoning.llm_router.settings") as mock_settings:
        mock_settings.gemini_api_key = ""
        mock_settings.deepseek_api_key = ""
        mock_settings.openrouter_api_key = "key"

        client = AsyncMock()
        client.complete = AsyncMock(return_value="openrouter answer")

        with patch("reasoning.llm_router._get_client", return_value=client):
            answer, provider, model = await ask(TaskType.CODE_QA, "question")

        assert answer == "openrouter answer"
        assert provider == Provider.OPENROUTER


@pytest.mark.asyncio
async def test_stream_ask_yields_chunks(all_keys_set):
    async def stream(*args, **kwargs):
        yield "Hello "
        yield "world"

    client = MagicMock()
    client.stream_complete = stream

    chunks = []
    with patch("reasoning.llm_router._get_client", return_value=client):
        async for chunk, provider, model in stream_ask(TaskType.CODE_QA, "q"):
            chunks.append(chunk)

    assert "Hello " in chunks
    assert "world" in chunks


@pytest.mark.asyncio
async def test_stream_ask_first_chunk_includes_provider(all_keys_set):
    async def stream(*args, **kwargs):
        yield "first"
        yield "second"

    client = MagicMock()
    client.stream_complete = stream

    with patch("reasoning.llm_router._get_client", return_value=client):
        first = True
        async for chunk, provider, model in stream_ask(TaskType.CODE_QA, "q"):
            if first:
                assert isinstance(provider, Provider)
                assert provider.value != ""
                first = False
            else:
                assert provider is None