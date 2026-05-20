"""Unit tests for reasoning.rag_chain."""

import pytest
from unittest.mock import AsyncMock, patch

from reasoning.context_assembler import AssembledContext
from reasoning.llm_router import Provider, TaskType
from reasoning.prompt_templates import BuiltPrompt, ContextChunkForPrompt, GraphContextForPrompt
from reasoning.rag_chain import RAGChain


@pytest.fixture
def mock_context() -> AssembledContext:
    chunk = ContextChunkForPrompt(
        file_path="auth/service.py",
        name="verify_password",
        display_name="AuthService.verify_password",
        chunk_type="function",
        start_line=45,
        end_line=67,
        score=0.934,
        content="def verify_password(): ...",
        docstring="Verify password.",
        language="python",
    )
    return AssembledContext(
        question="How does login work?",
        repo_id="repo-uuid-123",
        repo_name="my-app",
        chunks=[chunk],
        graph=GraphContextForPrompt(call_chain=["login", "verify_password"]),
        total_chunks_found=8,
        chunks_used=1,
        estimated_tokens=700,
        vector_search_ms=12.3,
        graph_expansion_ms=8.7,
        include_graph=True,
        top_result_score=0.934,
    )


@pytest.fixture
def mock_prompt() -> BuiltPrompt:
    return BuiltPrompt(
        system_prompt="You are a code expert.",
        user_prompt="<context>code</context>",
        task_type=TaskType.CODE_QA,
        template_version="1.0.0",
        estimated_tokens=850,
    )


@pytest.mark.asyncio
async def test_answer_returns_rag_response(mock_context, mock_prompt):
    chain = RAGChain()

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch(
                "reasoning.rag_chain.ask",
                AsyncMock(return_value=("Password uses bcrypt", Provider.DEEPSEEK, "deepseek-coder")),
            ):
                response = await chain.answer(repo_id="repo-uuid-123", question="How does login work?")

    assert response.answer == "Password uses bcrypt"
    assert response.provider_used == "deepseek"
    assert response.model_used == "deepseek-coder"


@pytest.mark.asyncio
async def test_answer_includes_citations(mock_context, mock_prompt):
    chain = RAGChain()

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch(
                "reasoning.rag_chain.ask",
                AsyncMock(return_value=("answer", Provider.GEMINI, "gemini-2.0-flash")),
            ):
                response = await chain.answer(repo_id="r", question="test")

    assert len(response.sources) == 1
    assert response.sources[0].file_path == "auth/service.py"
    assert response.sources[0].function_name == "AuthService.verify_password"


@pytest.mark.asyncio
async def test_answer_includes_graph_path(mock_context, mock_prompt):
    chain = RAGChain()

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch(
                "reasoning.rag_chain.ask",
                AsyncMock(return_value=("answer", Provider.GEMINI, "gemini-2.0-flash")),
            ):
                response = await chain.answer(repo_id="r", question="test")

    assert response.graph_path == ["login", "verify_password"]


@pytest.mark.asyncio
async def test_answer_auto_detects_task_type(mock_context, mock_prompt):
    chain = RAGChain()

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt) as build_prompt_mock:
            with patch(
                "reasoning.rag_chain.ask",
                AsyncMock(return_value=("answer", Provider.DEEPSEEK, "deepseek-coder")),
            ):
                await chain.answer(repo_id="r", question="Find security vulnerabilities in auth.py")

    kwargs = build_prompt_mock.call_args.kwargs
    assert kwargs["task_type"] == TaskType.SECURITY


@pytest.mark.asyncio
async def test_answer_has_timing_data(mock_context, mock_prompt):
    chain = RAGChain()

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch(
                "reasoning.rag_chain.ask",
                AsyncMock(return_value=("answer", Provider.GEMINI, "gemini-2.0-flash")),
            ):
                response = await chain.answer(repo_id="r", question="test")

    assert response.vector_search_ms >= 0
    assert response.total_latency_ms > 0


@pytest.mark.asyncio
async def test_stream_answer_yields_metadata_first(mock_context, mock_prompt):
    chain = RAGChain()

    async def fake_stream(*args, **kwargs):
        yield "Hello", Provider.GEMINI, "gemini-2.0-flash"
        yield " world", None, ""

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch("reasoning.rag_chain.stream_ask", side_effect=fake_stream):
                events = []
                async for event in chain.stream_answer(repo_id="r", question="test"):
                    events.append(event)

    assert events[0]["type"] == "metadata"
    assert events[1]["type"] == "delta"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_metadata_contains_sources_and_graph(mock_context, mock_prompt):
    chain = RAGChain()

    async def fake_stream(*args, **kwargs):
        yield "chunk", Provider.DEEPSEEK, "deepseek-coder"

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch("reasoning.rag_chain.stream_ask", side_effect=fake_stream):
                first = None
                async for event in chain.stream_answer(repo_id="r", question="test"):
                    first = event
                    break

    assert first is not None
    assert first["type"] == "metadata"
    assert "sources" in first
    assert "graph_path" in first


@pytest.mark.asyncio
async def test_stream_answer_handles_errors(mock_context, mock_prompt):
    chain = RAGChain()

    async def fake_stream(*args, **kwargs):
        raise RuntimeError("stream failed")
        yield  # pragma: no cover

    with patch.object(chain._assembler, "assemble", AsyncMock(return_value=mock_context)):
        with patch("reasoning.rag_chain.build_prompt", return_value=mock_prompt):
            with patch("reasoning.rag_chain.stream_ask", side_effect=fake_stream):
                events = []
                async for event in chain.stream_answer(repo_id="r", question="test"):
                    events.append(event)

    assert any(event["type"] == "error" for event in events)
    assert events[-1]["type"] == "done"
