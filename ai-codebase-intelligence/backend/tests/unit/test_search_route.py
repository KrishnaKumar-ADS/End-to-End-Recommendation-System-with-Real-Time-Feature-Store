"""Unit tests for api.routes.search endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.search import router
from db.database import get_db
from db.models import IngestionStatus


class _ScalarResult:
    def __init__(self, repo):
        self._repo = repo

    def scalar_one_or_none(self):
        return self._repo


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_search_returns_404_for_missing_repo(app: FastAPI):
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(None))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/search", params={"q": "hash", "repo_id": "11111111-1111-1111-1111-111111111111"})

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_returns_422_for_non_completed_repo(app: FastAPI):
    repo = SimpleNamespace(id="1", name="repo", status=IngestionStatus.PARSING)
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/search", params={"q": "hash", "repo_id": "11111111-1111-1111-1111-111111111111"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_success_shape(app: FastAPI):
    repo = SimpleNamespace(id="1", name="repo", status=IngestionStatus.COMPLETED)
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(repo))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    fake_response = SimpleNamespace(
        query="hash",
        expanded_queries=["hash"],
        repo_id="11111111-1111-1111-1111-111111111111",
        mode="hybrid",
        reranked=False,
        total_results=1,
        results=[
            SimpleNamespace(
                id="c1",
                name="hash_password",
                file_path="auth.py",
                chunk_type="function",
                start_line=1,
                end_line=2,
                content="def hash_password()",
                docstring=None,
                language="python",
                parent_name=None,
                vector_score=0.9,
                bm25_score=3.0,
                hybrid_score=0.02,
                rerank_score=0.02,
                vector_rank=1,
                bm25_rank=1,
                hybrid_rank=1,
                final_rank=1,
            )
        ],
        timing=SimpleNamespace(embed_ms=1, expand_ms=0, vector_ms=1, bm25_ms=1, fusion_ms=1, rerank_ms=0, total_ms=4),
    )

    with patch("api.routes.search.run_search", AsyncMock(return_value=fake_response)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/search", params={"q": "hash", "repo_id": "11111111-1111-1111-1111-111111111111"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "hybrid"
    assert body["results"][0]["name"] == "hash_password"
