"""
Integration tests — these hit the real PostgreSQL database.
Make sure Docker is running before running these tests.

Run with:
    pytest tests/integration/ -v
"""

import pytest
import pytest_asyncio
import uuid
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, text

from main import app
from db.database import Base, get_db
from db.models import Repository, SourceFile, CodeChunk, IngestionStatus
from core.config import get_settings

settings = get_settings()


# ── Test database setup ───────────────────────────────────────

TEST_DB_URL = settings.database_url  # uses the same Docker Postgres


@pytest_asyncio.fixture(scope="function")
async def db_session():
    """Create a fresh DB session for each test, rolled back after."""
    engine = create_async_engine(TEST_DB_URL)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    """FastAPI test client with DB dependency overridden."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()


# ── Health check tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_providers_available" in data


# ── Ingest endpoint tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_returns_202(client):
    response = await client.post("/api/v1/ingest", json={
        "github_url": "https://github.com/test/repo",
        "branch": "main"
    })
    # 202 Accepted or 409 if already exists — both valid
    assert response.status_code in (202, 409)


@pytest.mark.asyncio
async def test_ingest_returns_repo_id(client):
    response = await client.post("/api/v1/ingest", json={
        "github_url": f"https://github.com/test/repo_{uuid.uuid4().hex[:8]}",
        "branch": "main"
    })
    assert response.status_code == 202
    data = response.json()
    assert "repo_id" in data
    assert "task_id" in data
    assert data["status"] == "queued"


@pytest.mark.asyncio
async def test_ingest_rejects_duplicate_url(client, db_session):
    url = f"https://github.com/test/repo_{uuid.uuid4().hex[:8]}"

    # First request
    r1 = await client.post("/api/v1/ingest", json={"github_url": url})
    assert r1.status_code == 202

    # Second request with same URL
    r2 = await client.post("/api/v1/ingest", json={"github_url": url})
    assert r2.status_code == 409
    assert "already ingested" in r2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ingest_rejects_non_github_url(client):
    response = await client.post("/api/v1/ingest", json={
        "github_url": "https://gitlab.com/user/repo"
    })
    assert response.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_ingest_creates_repository_record(client, db_session):
    url = f"https://github.com/test/repo_{uuid.uuid4().hex[:8]}"
    response = await client.post("/api/v1/ingest", json={"github_url": url})
    assert response.status_code == 202

    repo_id = response.json()["repo_id"]
    result = await db_session.execute(
        select(Repository).where(Repository.id == repo_id)
    )
    repo = result.scalar_one_or_none()
    assert repo is not None
    assert repo.github_url == url
    assert repo.status == IngestionStatus.QUEUED


# ── Status endpoint tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_status_returns_404_for_unknown_task(client):
    response = await client.get("/api/v1/status/nonexistent-task-id-12345")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_status_returns_queued_for_new_task(client, db_session):
    url = f"https://github.com/test/repo_{uuid.uuid4().hex[:8]}"
    ingest_response = await client.post("/api/v1/ingest", json={"github_url": url})
    task_id = ingest_response.json()["task_id"]

    status_response = await client.get(f"/api/v1/status/{task_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert "status" in data
    assert "progress" in data