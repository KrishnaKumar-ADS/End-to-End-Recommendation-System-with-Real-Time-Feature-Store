"""Integration tests for Week 10 ask pipeline behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.ask import router as ask_router
from core.exceptions import LLMProviderError
from db.database import get_db
from db.models import IngestionStatus
from semantic_cache.models import CachedAnswer, CacheLookupResult


class _ScalarResult:
    def __init__(self, repo):
        self._repo = repo

    def scalar_one_or_none(self):
        return self._repo


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(ask_router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_ask_returns_quality_score_on_happy_path(app: FastAPI):
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

    source = SimpleNamespace(
        file_path="auth.py",
        function_name="verify_password",
        start_line=10,
        end_line=20,
        score=0.91,
        chunk_type="function",
    )
    rag_response = SimpleNamespace(
        request_id="abcd1234",
        answer="Authentication uses bcrypt.verify_password in auth.py.",
        provider_used="openrouter",
        model_used="qwen/qwen-2.5-coder-32b-instruct",
        task_type="code_qa",
        sources=[source],
        graph_path=["login", "verify_password"],
        context_chunks_used=1,
        estimated_tokens=200,
        vector_search_ms=10.0,
        graph_expansion_ms=5.0,
        total_latency_ms=120.0,
        top_result_score=0.92,
    )

    quality = MagicMock()
    quality.model_dump.return_value = {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "completeness": 0.7,
        "overall": 0.82,
        "critique": "Grounded.",
        "skipped": False,
        "skip_reason": None,
        "judge_model": "qwen/qwen-max",
    }

    with patch("api.routes.ask._rag_chain.answer", AsyncMock(return_value=rag_response)):
        with patch("api.routes.ask._quality_evaluator.score", AsyncMock(return_value=quality)):
            with patch("api.routes.ask._semantic_cache.lookup", AsyncMock(return_value=CacheLookupResult(found=False))):
                with patch("api.routes.ask._semantic_cache.store", AsyncMock(return_value=None)):
                    with patch("embeddings.gemini_embedder.embed_query", return_value=[0.1] * 768):
                        with patch("api.routes.ask._cost_tracker.record", AsyncMock(return_value=0.001)):
                            transport = ASGITransport(app=app)
                            async with AsyncClient(transport=transport, base_url="http://test") as client:
                                response = await client.post(
                                    "/api/v1/ask",
                                    json={
                                        "repo_id": "11111111-1111-1111-1111-111111111111",
                                        "question": "How does authentication work?",
                                    },
                                )

    assert response.status_code == 200
    body = response.json()
    assert "quality_score" in body
    assert body["cached"] is False


@pytest.mark.asyncio
async def test_ask_returns_cached_answer_on_semantic_hit(app: FastAPI):
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

    cached_answer = CachedAnswer(
        answer="Cached authentication answer",
        sources=[{"file_path": "auth.py", "function_name": "verify_password"}],
        quality_score={"overall": 0.88},
        repo_id="11111111-1111-1111-1111-111111111111",
        provider_used="openrouter",
        model_used="qwen/qwen-2.5-coder-32b-instruct",
    )

    with patch("api.routes.ask._semantic_cache.lookup", AsyncMock(return_value=CacheLookupResult(found=True, similarity=0.95, cached_answer=cached_answer))):
        with patch("embeddings.gemini_embedder.embed_query", return_value=[0.1] * 768):
            with patch("api.routes.ask._rag_chain.answer", AsyncMock()) as rag_mock:
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/ask",
                        json={
                            "repo_id": "11111111-1111-1111-1111-111111111111",
                            "question": "Explain auth flow",
                        },
                    )

    assert response.status_code == 200
    body = response.json()
    assert body["cached"] is True
    assert body["cache_similarity"] == 0.95
    rag_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_returns_404_when_repo_missing(app: FastAPI):
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
async def test_ask_returns_422_when_question_missing(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ask",
            json={"repo_id": "11111111-1111-1111-1111-111111111111"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ask_returns_fallback_response_when_llm_providers_fail(app: FastAPI):
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

    with patch("api.routes.ask._semantic_cache.lookup", AsyncMock(return_value=CacheLookupResult(found=False))):
        with patch("api.routes.ask._rag_chain.answer", AsyncMock(side_effect=LLMProviderError("All failed"))):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/ask",
                    json={
                        "repo_id": "11111111-1111-1111-1111-111111111111",
                        "question": "How does auth work?",
                    },
                )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider_used"] == "fallback"
    assert payload["model_used"] == "retrieval-only"
    assert payload["quality_score"]["skipped"] is True
