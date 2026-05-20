"""Unit tests for reasoning.deepseek_client with mocked AsyncOpenAI calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.exceptions import LLMProviderError
from reasoning.deepseek_client import DeepSeekClient


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
def deepseek_key():
    with patch("reasoning.deepseek_client.settings") as settings_mock:
        settings_mock.deepseek_api_key = "key"
        yield settings_mock


def test_init_raises_without_key():
    with patch("reasoning.deepseek_client.settings") as settings_mock:
        settings_mock.deepseek_api_key = ""
        with pytest.raises(LLMProviderError, match="DEEPSEEK_API_KEY"):
            DeepSeekClient()


def test_init_creates_async_client(deepseek_key):
    with patch("reasoning.deepseek_client.AsyncOpenAI") as client_cls:
        DeepSeekClient(model="deepseek-coder")
    client_cls.assert_called_once()


def test_get_context_window_for_models(deepseek_key):
    with patch("reasoning.deepseek_client.AsyncOpenAI"):
        client = DeepSeekClient(model="deepseek-coder")
    assert client.get_context_window("deepseek-coder") == 128_000
    assert client.get_context_window("deepseek-chat") == 65_536


def test_build_messages_with_system_prompt():
    messages = DeepSeekClient._build_messages("user", "system")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"


def test_build_messages_without_system_prompt():
    messages = DeepSeekClient._build_messages("user", "")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


@pytest.mark.asyncio
async def test_complete_returns_message_content(deepseek_key):
    with patch("reasoning.deepseek_client.AsyncOpenAI"):
        client = DeepSeekClient(model="deepseek-coder")

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    with patch.object(client, "_complete_with_retry", AsyncMock(return_value=response)):
        text = await client.complete("prompt")

    assert text == "answer"


@pytest.mark.asyncio
async def test_complete_wraps_errors(deepseek_key):
    with patch("reasoning.deepseek_client.AsyncOpenAI"):
        client = DeepSeekClient(model="deepseek-coder")

    with patch.object(client, "_complete_with_retry", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(LLMProviderError, match="DeepSeek generation failed"):
            await client.complete("prompt")


@pytest.mark.asyncio
async def test_stream_complete_yields_deltas(deepseek_key):
    client_obj = MagicMock()
    chunk1 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))])
    chunk2 = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))])
    client_obj.chat.completions.create = AsyncMock(return_value=_AsyncStream([chunk1, chunk2]))

    with patch("reasoning.deepseek_client.AsyncOpenAI", return_value=client_obj):
        client = DeepSeekClient(model="deepseek-coder")
        received = []
        async for part in client.stream_complete("prompt"):
            received.append(part)

    assert received == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_stream_complete_wraps_errors(deepseek_key):
    client_obj = MagicMock()
    client_obj.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("reasoning.deepseek_client.AsyncOpenAI", return_value=client_obj):
        client = DeepSeekClient(model="deepseek-coder")
        with pytest.raises(LLMProviderError, match="DeepSeek streaming failed"):
            async for _ in client.stream_complete("prompt"):
                pass
