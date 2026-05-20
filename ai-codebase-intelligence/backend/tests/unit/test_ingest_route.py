"""Unit tests for ingest route repository detail and deletion behaviors."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.ingest import router
from db.database import get_db


class _ScalarCount:
    def __init__(self, value: int):
        self._value = value

    def scalar(self):
        return self._value


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_delete_repository_returns_422_for_invalid_id(app: FastAPI):
    fake_db = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/api/v1/repos/not-a-uuid")

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_delete_repository_returns_404_when_missing(app: FastAPI):
    fake_db = AsyncMock()
    fake_db.get = AsyncMock(return_value=None)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/api/v1/repos/11111111-1111-1111-1111-111111111111")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_repository_success(app: FastAPI):
    repo_id = "11111111-1111-1111-1111-111111111111"
    repo = SimpleNamespace(id=repo_id, name="demo-repo")

    fake_db = AsyncMock()
    fake_db.get = AsyncMock(return_value=repo)
    fake_db.execute = AsyncMock(side_effect=[_ScalarCount(5), _ScalarCount(12)])
    fake_db.delete = AsyncMock()
    fake_db.commit = AsyncMock()
    fake_db.rollback = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    cache_manager = SimpleNamespace(invalidate_repo=Mock(return_value=4))
    semantic_cache = SimpleNamespace(invalidate_repo=AsyncMock(return_value=2))

    with (
        patch("embeddings.vector_store.count_repo_vectors", return_value=9) as mock_count_vectors,
        patch("embeddings.vector_store.delete_repo_vectors") as mock_delete_vectors,
        patch("graph.neo4j_writer.delete_repo_graph", return_value=7) as mock_delete_graph,
        patch("caching.cache_manager.get_cache_manager", return_value=cache_manager),
        patch("semantic_cache.answer_cache.get_semantic_answer_cache", return_value=semantic_cache),
        patch("ingestion.repo_loader.delete_repository") as mock_delete_raw,
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete(f"/api/v1/repos/{repo_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["repo_id"] == repo_id
    assert body["repo_name"] == "demo-repo"
    assert body["deleted_files"] == 5
    assert body["deleted_chunks"] == 12
    assert body["deleted_vectors"] == 9
    assert body["deleted_graph_nodes"] == 7
    assert body["deleted_cache_keys"] == 6
    assert body["warnings"] == []

    mock_count_vectors.assert_called_once_with(repo_id)
    mock_delete_vectors.assert_called_once_with(repo_id)
    mock_delete_graph.assert_called_once_with(repo_id)
    semantic_cache.invalidate_repo.assert_awaited_once_with(repo_id)
    mock_delete_raw.assert_called_once_with(repo_id)
    fake_db.delete.assert_awaited_once_with(repo)
    fake_db.commit.assert_awaited_once()
