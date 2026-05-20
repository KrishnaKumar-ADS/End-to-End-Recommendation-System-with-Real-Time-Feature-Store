"""Unit tests for reasoning.openrouter_client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import LLMProviderError
from reasoning.openrouter_client import DEFAULT_MODEL, OPENROUTER_BASE_URL, OPENROUTER_HEADERS, OpenRouterClient


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


@pytest.fixture
def openrouter_key():
    with patch("reasoning.openrouter_client.settings") as settings_mock:
        settings_mock.openrouter_api_key = "key"
        yield settings_mock


def test_init_raises_without_key():
    with patch("reasoning.openrouter_client.settings") as settings_mock:
        settings_mock.openrouter_api_key = ""
        with pytest.raises(LLMProviderError, match="OPENROUTER_API_KEY"):
            OpenRouterClient()


def test_init_sets_base_url_and_headers(openrouter_key):
    with patch("reasoning.openrouter_client.AsyncOpenAI") as client_cls:
        OpenRouterClient()
    kwargs = client_cls.call_args.kwargs
    assert kwargs["base_url"] == OPENROUTER_BASE_URL
    assert kwargs["default_headers"] == OPENROUTER_HEADERS


def test_default_model_is_free_llama():
    assert DEFAULT_MODEL == "openrouter/free"


@pytest.mark.asyncio
async def test_complete_returns_content(openrouter_key):
    with patch("reasoning.openrouter_client.AsyncOpenAI"):
        client = OpenRouterClient()

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )
    with patch.object(client, "_complete_with_retry", AsyncMock(return_value=response)):
        text = await client.complete("prompt")

    assert text == "answer"


@pytest.mark.asyncio
async def test_stream_complete_yields_chunks(openrouter_key):
    client_obj = MagicMock()
    chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi "))])
    chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="there"))])
    client_obj.chat.completions.create = AsyncMock(return_value=_AsyncStream([chunk1, chunk2]))

    with patch("reasoning.openrouter_client.AsyncOpenAI", return_value=client_obj):
        client = OpenRouterClient()
        parts = []
        async for part in client.stream_complete("prompt"):
            parts.append(part)

    assert parts == ["Hi ", "there"]


@pytest.mark.asyncio
async def test_list_free_models_filters_prompt_price_zero(openrouter_key):
    payload = {
        "data": [
            {"id": "free-model", "pricing": {"prompt": "0"}},
            {"id": "paid-model", "pricing": {"prompt": "0.0001"}},
        ]
    }

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = payload

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=mock_response)
    mock_http.__aenter__.return_value = mock_http
    mock_http.__aexit__.return_value = None

    with patch("reasoning.openrouter_client.AsyncOpenAI"):
        client = OpenRouterClient()
    with patch("reasoning.openrouter_client.httpx.AsyncClient", return_value=mock_http):
        models = await client.list_free_models()

    assert models == ["free-model"]


@pytest.mark.asyncio
async def test_list_free_models_fallback_on_http_failure(openrouter_key):
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=RuntimeError("network"))
    mock_http.__aenter__.return_value = mock_http
    mock_http.__aexit__.return_value = None

    with patch("reasoning.openrouter_client.AsyncOpenAI"):
        client = OpenRouterClient()
    with patch("reasoning.openrouter_client.httpx.AsyncClient", return_value=mock_http):
        models = await client.list_free_models()

    assert models == [DEFAULT_MODEL]
