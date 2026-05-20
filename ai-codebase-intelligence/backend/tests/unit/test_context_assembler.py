"""Unit tests for reasoning.context_assembler."""

from unittest.mock import AsyncMock, patch

import pytest

from reasoning.context_assembler import (
    MIN_CHUNK_SCORE,
    ContextAssembler,
    ContextAssemblerConfig,
)
from reasoning.prompt_templates import GraphContextForPrompt


@pytest.fixture
def assembler() -> ContextAssembler:
    return ContextAssembler()


@pytest.mark.asyncio
async def test_assemble_calls_embed_then_search(assembler):
    with patch.object(assembler, "_embed_question", AsyncMock(return_value=[0.1] * 768)) as embed_mock:
        with patch.object(
            assembler,
            "_vector_search",
            AsyncMock(
                return_value=[
                    {
                        "chunk_id": "c1",
                        "name": "fn",
                        "display_name": "fn",
                        "file_path": "a.py",
                        "language": "python",
                        "chunk_type": "function",
                        "content": "def fn():\n    pass",
                        "start_line": 1,
                        "end_line": 2,
                        "docstring": "",
                        "score": 0.9,
                    }
                ]
            ),
        ) as search_mock:
            with patch.object(assembler, "_expand_graph", AsyncMock(return_value=GraphContextForPrompt())):
                context = await assembler.assemble("repo", "question", ContextAssemblerConfig(include_graph=True))

    embed_mock.assert_awaited_once()
    search_mock.assert_awaited_once()
    assert context.chunks_used == 1


@pytest.mark.asyncio
async def test_assemble_skips_graph_when_disabled(assembler):
    with patch.object(assembler, "_embed_question", AsyncMock(return_value=[0.1] * 768)):
        with patch.object(
            assembler,
            "_vector_search",
            AsyncMock(
                return_value=[
                    {
                        "chunk_id": "c1",
                        "name": "fn",
                        "display_name": "fn",
                        "file_path": "a.py",
                        "language": "python",
                        "chunk_type": "function",
                        "content": "def fn():\n    pass",
                        "start_line": 1,
                        "end_line": 2,
                        "docstring": "",
                        "score": 0.9,
                    }
                ]
            ),
        ):
            with patch.object(assembler, "_expand_graph", AsyncMock()) as graph_mock:
                await assembler.assemble("repo", "question", ContextAssemblerConfig(include_graph=False))

    graph_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_assemble_discards_low_scores(assembler):
    with patch.object(assembler, "_embed_question", AsyncMock(return_value=[0.1] * 768)):
        with patch.object(
            assembler,
            "_vector_search",
            AsyncMock(
                return_value=[
                    {
                        "chunk_id": "c1",
                        "name": "low",
                        "display_name": "low",
                        "file_path": "a.py",
                        "language": "python",
                        "chunk_type": "function",
                        "content": "low",
                        "start_line": 1,
                        "end_line": 1,
                        "docstring": "",
                        "score": MIN_CHUNK_SCORE - 0.01,
                    },
                    {
                        "chunk_id": "c2",
                        "name": "high",
                        "display_name": "high",
                        "file_path": "b.py",
                        "language": "python",
                        "chunk_type": "function",
                        "content": "high",
                        "start_line": 2,
                        "end_line": 2,
                        "docstring": "",
                        "score": MIN_CHUNK_SCORE + 0.1,
                    },
                ]
            ),
        ):
            with patch.object(assembler, "_expand_graph", AsyncMock(return_value=GraphContextForPrompt())):
                context = await assembler.assemble("repo", "question")

    assert context.chunks_used == 1
    assert context.chunks[0].display_name == "high"


def test_apply_token_budget_keeps_first_even_if_large(assembler):
    results = [
        {"content": "x" * 10000, "name": "a"},
        {"content": "x" * 10000, "name": "b"},
    ]
    selected, total = assembler._apply_token_budget(results, available_tokens=1)
    assert len(selected) == 1
    assert total > 0


def test_apply_token_budget_stops_when_exhausted(assembler):
    results = [
        {"content": "x" * 40},
        {"content": "x" * 40},
        {"content": "x" * 40},
    ]
    selected, total = assembler._apply_token_budget(results, available_tokens=15)
    assert len(selected) < len(results)


def test_to_prompt_chunk_maps_fields(assembler):
    result = {
        "file_path": "a.py",
        "name": "foo",
        "display_name": "foo",
        "chunk_type": "function",
        "start_line": 10,
        "end_line": 20,
        "score": 0.88,
        "content": "def foo(): pass",
        "docstring": "doc",
        "language": "python",
    }
    chunk = assembler._to_prompt_chunk(result)
    assert chunk.file_path == "a.py"
    assert chunk.start_line == 10
    assert chunk.score == 0.88


@pytest.mark.asyncio
async def test_expand_graph_returns_empty_on_failure(assembler):
    with patch("graph.graph_utils.get_subgraph", side_effect=RuntimeError("neo4j down")):
        graph = await assembler._expand_graph("repo", {"chunk_id": "c1"}, hops=2)
    assert isinstance(graph, GraphContextForPrompt)
    assert graph.call_chain == []


@pytest.mark.asyncio
async def test_assemble_records_vector_search_timing(assembler):
    with patch.object(assembler, "_embed_question", AsyncMock(return_value=[0.1] * 768)):
        with patch.object(assembler, "_vector_search", AsyncMock(return_value=[])):
            context = await assembler.assemble("repo", "question")
    assert context.vector_search_ms >= 0
