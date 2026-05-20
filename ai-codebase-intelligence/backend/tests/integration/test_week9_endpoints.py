"""Integration smoke tests for Week 9 endpoints: cache and explain."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.cache import router as cache_router
from api.routes.explain import router as explain_router
from core.exceptions import ChunkNotFoundError
from db.database import get_db
from explanation.schemas import ExplainResponse, ReturnInfo


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    app.include_router(explain_router)
    return app


@pytest.mark.asyncio
async def test_cache_stats_returns_payload(app: FastAPI):
    fake_cache = MagicMock()
    fake_cache.get_stats.return_value = {
        "embedding": {"hits": 10, "misses": 2, "total": 12, "hit_rate": 0.833, "key_count": 3},
        "search": {"hits": 5, "misses": 5, "total": 10, "hit_rate": 0.5, "key_count": 2},
        "graph": {"hits": 9, "misses": 1, "total": 10, "hit_rate": 0.9, "key_count": 4},
        "redis_memory_bytes": 1024,
    }

    with patch("api.routes.cache.get_cache_manager", return_value=fake_cache):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/cache/stats")

    assert response.status_code == 200
    body = response.json()
    assert "embedding" in body
    assert "search" in body
    assert "graph" in body


@pytest.mark.asyncio
async def test_cache_invalidate_returns_deleted_count(app: FastAPI):
    fake_cache = MagicMock()
    fake_cache.invalidate_repo.return_value = 7
    fake_semantic = MagicMock()
    fake_semantic.invalidate_repo = AsyncMock(return_value=0)

    with patch("api.routes.cache.get_cache_manager", return_value=fake_cache):
        with patch("api.routes.cache.get_semantic_answer_cache", return_value=fake_semantic):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete("/api/v1/cache/repo-123")

    assert response.status_code == 200
    body = response.json()
    assert body["repo_id"] == "repo-123"
    assert body["keys_deleted"] == 7


@pytest.mark.asyncio
async def test_explain_requires_function_name_or_chunk_id(app: FastAPI):
    fake_db = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/explain",
            json={"repo_id": "11111111-1111-1111-1111-111111111111"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_explain_success_returns_structured_response(app: FastAPI):
    fake_db = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    expected = ExplainResponse(
        function_name="verify_password",
        file_path="auth/service.py",
        start_line=45,
        end_line=67,
        summary="Verifies password using bcrypt.",
        parameters=[],
        returns=ReturnInfo(type_annotation="bool", description="True when credentials are valid."),
        side_effects=[],
        callers=[],
        callees=[],
        complexity_score=3,
        provider_used="openrouter",
        model_used="qwen/qwen-2.5-coder-32b-instruct",
        explanation_ms=12.4,
    )

    service = AsyncMock()
    service.explain = AsyncMock(return_value=expected)

    with patch("api.routes.explain.CodeExplainer", return_value=service):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/explain",
                json={
                    "repo_id": "11111111-1111-1111-1111-111111111111",
                    "function_name": "verify_password",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["function_name"] == "verify_password"
    assert body["file_path"] == "auth/service.py"
    assert body["complexity_score"] == 3


@pytest.mark.asyncio
async def test_explain_returns_404_for_missing_chunk(app: FastAPI):
    fake_db = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    service = AsyncMock()
    service.explain = AsyncMock(side_effect=ChunkNotFoundError("Missing"))

    with patch("api.routes.explain.CodeExplainer", return_value=service):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/explain",
                json={
                    "repo_id": "11111111-1111-1111-1111-111111111111",
                    "function_name": "verify_password",
                },
            )

    assert response.status_code == 404