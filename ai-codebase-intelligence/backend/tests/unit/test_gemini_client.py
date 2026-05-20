"""Unit tests for reasoning.gemini_client with mocked Gemini SDK calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.exceptions import LLMProviderError
from reasoning.gemini_client import GEMINI_FLASH, GeminiChatClient


@pytest.fixture
def gemini_key():
    with patch("reasoning.gemini_client.settings") as settings_mock:
        settings_mock.gemini_api_key = "key"
        yield settings_mock


def test_init_raises_without_key():
    with patch("reasoning.gemini_client.settings") as settings_mock:
        settings_mock.gemini_api_key = ""
        with pytest.raises(LLMProviderError, match="GEMINI_API_KEY"):
            GeminiChatClient()


def test_init_configures_model(gemini_key):
    with patch("reasoning.gemini_client.genai.configure") as configure_mock:
        with patch("reasoning.gemini_client.genai.GenerativeModel") as model_mock:
            GeminiChatClient(model="gemini-2.0-flash")

    configure_mock.assert_called_once()
    model_mock.assert_called_once_with("gemini-2.0-flash")


def test_build_contents_with_system_prompt(gemini_key):
    with patch("reasoning.gemini_client.genai.GenerativeModel"):
        client = GeminiChatClient()
    contents = client._build_contents("user prompt", "system prompt")
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[-1]["parts"][0]["text"] == "user prompt"


def test_build_contents_without_system_prompt(gemini_key):
    with patch("reasoning.gemini_client.genai.GenerativeModel"):
        client = GeminiChatClient()
    contents = client._build_contents("user prompt", "")
    assert len(contents) == 1
    assert contents[0]["role"] == "user"


def test_context_window_default(gemini_key):
    with patch("reasoning.gemini_client.genai.GenerativeModel"):
        client = GeminiChatClient(model=GEMINI_FLASH)
    assert client.context_window_tokens == 1_048_576


@pytest.mark.asyncio
async def test_complete_returns_text(gemini_key):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = SimpleNamespace(text="hello")

    with patch("reasoning.gemini_client.genai.GenerativeModel", return_value=mock_model):
        client = GeminiChatClient()
        text = await client.complete("prompt")

    assert text == "hello"


@pytest.mark.asyncio
async def test_complete_raises_on_empty_text(gemini_key):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = SimpleNamespace(text="")

    with patch("reasoning.gemini_client.genai.GenerativeModel", return_value=mock_model):
        client = GeminiChatClient()
        with pytest.raises(LLMProviderError, match="empty response"):
            await client.complete("prompt")


@pytest.mark.asyncio
async def test_stream_complete_yields_chunks(gemini_key):
    chunks = [SimpleNamespace(text="Hello "), SimpleNamespace(text="world")]
    mock_model = MagicMock()
    mock_model.generate_content.return_value = chunks

    with patch("reasoning.gemini_client.genai.GenerativeModel", return_value=mock_model):
        client = GeminiChatClient()
        received = []
        async for chunk in client.stream_complete("prompt"):
            received.append(chunk)

    assert received == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_stream_complete_propagates_error(gemini_key):
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("boom")

    with patch("reasoning.gemini_client.genai.GenerativeModel", return_value=mock_model):
        client = GeminiChatClient()
        with pytest.raises(LLMProviderError, match="Gemini streaming failed"):
            async for _ in client.stream_complete("prompt"):
                pass


def test_estimate_tokens_falls_back_without_tiktoken(gemini_key):
    with patch("reasoning.gemini_client.genai.GenerativeModel"):
        client = GeminiChatClient()

    with patch.dict("sys.modules", {"tiktoken": None}):
        tokens = client.estimate_tokens("abcd" * 10)

    assert tokens > 0
