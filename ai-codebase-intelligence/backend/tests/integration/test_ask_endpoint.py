"""Integration-style tests for api.routes.ask with mocked dependencies."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.ask import router as ask_router
from db.database import get_db
from db.models import IngestionStatus
from reasoning.llm_router import Provider


class _ScalarResult:
    def __init__(self, repo):
        self._repo = repo

    def scalar_one_or_none(self):
        return self._repo


class _AsyncSessionCtx:
    async def __aenter__(self):
        return AsyncMock()

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(ask_router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_post_ask_returns_404_for_unknown_repo(app: FastAPI):
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(None))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "How does auth work?"},
        )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_post_ask_returns_422_when_repo_not_completed(app: FastAPI):
    repo = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="repo",
        status=IngestionStatus.PARSING,
    )
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "How does auth work?"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_ask_rejects_short_question(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "hey"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_ask_rejects_whitespace_question(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "       "},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_ask_non_stream_returns_json_body(app: FastAPI):
    repo = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="repo",
        status=IngestionStatus.COMPLETED,
    )
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    response_payload = SimpleNamespace(
        request_id="abcd1234",
        answer="Auth uses bcrypt.",
        provider_used="deepseek",
        model_used="deepseek-coder",
        task_type="code_qa",
        sources=[],
        graph_path=["login", "verify_password"],
        context_chunks_used=3,
        estimated_tokens=900,
        vector_search_ms=10.0,
        graph_expansion_ms=5.0,
        total_latency_ms=120.0,
        top_result_score=0.92,
    )

    with patch("api.routes.ask._rag_chain.answer", AsyncMock(return_value=response_payload)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ask",
                json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "How does auth work?", "stream": False},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Auth uses bcrypt."
    assert body["provider_used"] == "deepseek"


@pytest.mark.asyncio
async def test_post_ask_stream_returns_event_stream_and_done(app: FastAPI):
    repo = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="repo",
        status=IngestionStatus.COMPLETED,
    )
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    async def stream_events(*args, **kwargs):
        yield {"type": "metadata", "provider": "deepseek", "model": "deepseek-coder"}
        yield {"type": "delta", "delta": "Hello"}
        yield {"type": "done", "total_latency_ms": 100}

    with patch("api.routes.ask._rag_chain.stream_answer", side_effect=stream_events):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ask",
                json={"repo_id": "11111111-1111-1111-1111-111111111111", "question": "How does auth work?", "stream": True},
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_get_providers_lists_available(app: FastAPI):
    with patch("api.routes.ask.get_available_providers", return_value=[Provider.GEMINI, Provider.DEEPSEEK]):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/ask/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert "gemini" in body["providers"]
    assert "deepseek" in body["providers"]


@pytest.mark.asyncio
async def test_post_deep_ask_decomposes_and_synthesizes(app: FastAPI):
    repo = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="repo",
        status=IngestionStatus.COMPLETED,
    )
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    source1 = SimpleNamespace(file_path="auth.py", function_name="login", start_line=10, end_line=20)
    source2 = SimpleNamespace(file_path="security.py", function_name="audit", start_line=40, end_line=60)
    rag_response_1 = SimpleNamespace(answer="auth answer", sources=[source1], provider_used="openrouter")
    rag_response_2 = SimpleNamespace(answer="security answer", sources=[source2], provider_used="openrouter")

    with patch("api.routes.ask.QueryDecomposer.decompose", AsyncMock(return_value=["q1", "q2"])):
        with patch("api.routes.ask.IntentClassifier.classify", AsyncMock(side_effect=["architecture", "security"])):
            with patch("api.routes.ask._rag_chain.answer", AsyncMock(side_effect=[rag_response_1, rag_response_2])) as answer_mock:
                with patch("api.routes.ask.AnswerSynthesizer.synthesize", AsyncMock(return_value="final synthesized answer")):
                    with patch("api.routes.ask.AsyncSessionLocal", side_effect=lambda: _AsyncSessionCtx()):
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/api/v1/ask/deep",
                                json={
                                    "repo_id": "11111111-1111-1111-1111-111111111111",
                                    "question": "How does auth work and what are its security risks?",
                                },
                            )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "final synthesized answer"
    assert body["sub_questions"] == ["q1", "q2"]
    assert body["partial_answers"] == ["auth answer", "security answer"]
    assert body["decomposed"] is True
    assert body["providers_used"] == ["openrouter"]
    assert answer_mock.await_count == 2


@pytest.mark.asyncio
async def test_post_deep_ask_handles_sub_question_failure(app: FastAPI):
    repo = SimpleNamespace(
        id="11111111-1111-1111-1111-111111111111",
        name="repo",
        status=IngestionStatus.COMPLETED,
    )
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    source = SimpleNamespace(file_path="auth.py", function_name="verify", start_line=11, end_line=21)
    success_response = SimpleNamespace(answer="sub-answer", sources=[source], provider_used="deepseek")

    async def _answer_side_effect(*args, **kwargs):
        if kwargs.get("question") == "q1":
            raise RuntimeError("boom")
        return success_response

    with patch("api.routes.ask.QueryDecomposer.decompose", AsyncMock(return_value=["q1", "q2"])):
        with patch("api.routes.ask.IntentClassifier.classify", AsyncMock(return_value="general")):
            with patch("api.routes.ask._rag_chain.answer", AsyncMock(side_effect=_answer_side_effect)):
                with patch("api.routes.ask.AnswerSynthesizer.synthesize", AsyncMock(return_value="final")):
                    with patch("api.routes.ask.AsyncSessionLocal", side_effect=lambda: _AsyncSessionCtx()):
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/api/v1/ask/deep",
                                json={
                                    "repo_id": "11111111-1111-1111-1111-111111111111",
                                    "question": "Complex question",
                                },
                            )

    assert response.status_code == 200
    body = response.json()
    assert body["partial_answers"][0] == "Failed to answer sub-question: q1"
    assert body["partial_answers"][1] == "sub-answer"
    assert body["providers_used"] == ["deepseek"]
