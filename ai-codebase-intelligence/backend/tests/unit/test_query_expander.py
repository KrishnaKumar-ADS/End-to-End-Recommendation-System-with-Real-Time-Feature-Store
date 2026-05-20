"""Unit tests for search.query_expander (network calls mocked)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_expand_query_returns_original_when_no_keys():
    from search import query_expander as module

    with patch.object(module.settings, "openrouter_api_key", ""), patch.object(module.settings, "qwen_api_key", ""):
        output = await module.expand_query("password hashing")

    assert output == ["password hashing"]


@pytest.mark.asyncio
async def test_expand_query_parses_json_array():
    from search import query_expander as module

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=json.dumps(["variant one", "variant two", "variant three"])))]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(module.settings, "openrouter_api_key", "sk-or-test"), patch.object(module.settings, "qwen_api_key", ""):
        with patch("search.query_expander.AsyncOpenAI", return_value=mock_client):
            output = await module.expand_query("password hashing", n_expansions=2)

    assert output[0] == "password hashing"
    assert len(output) == 3


@pytest.mark.asyncio
async def test_expand_query_handles_markdown_fence():
    from search import query_expander as module

    content = "```json\n[\"v1\", \"v2\"]\n```"
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(module.settings, "openrouter_api_key", "sk-or-test"), patch.object(module.settings, "qwen_api_key", ""):
        with patch("search.query_expander.AsyncOpenAI", return_value=mock_client):
            output = await module.expand_query("base", n_expansions=2)

    assert output[0] == "base"
    assert "v1" in output


@pytest.mark.asyncio
async def test_expand_query_falls_back_on_parse_error():
    from search import query_expander as module

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="NOT JSON"))]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    with patch.object(module.settings, "openrouter_api_key", "sk-or-test"), patch.object(module.settings, "qwen_api_key", ""):
        with patch("search.query_expander.AsyncOpenAI", return_value=mock_client):
            output = await module.expand_query("base")

    assert output == ["base"]
