"""Integration tests for Week 10 endpoints: evaluate, metrics, webhook."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes.evaluate import router as evaluate_router
from api.routes.metrics import router as metrics_router
from api.routes.webhook import router as webhook_router
from cost_tracking.models import DailyCostSummary
from db.database import get_db
from reasoning.circuit_breaker import CircuitState


class _ScalarResult:
    def __init__(self, repo):
        self._repo = repo

    def scalar_one_or_none(self):
        return self._repo


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(evaluate_router)
    app.include_router(metrics_router)
    app.include_router(webhook_router)
    return app


@pytest.mark.asyncio
async def test_post_evaluate_returns_scores(app: FastAPI):
    score_obj = MagicMock()
    score_obj.model_dump.return_value = {
        "faithfulness": 0.9,
        "relevance": 0.8,
        "completeness": 0.7,
        "overall": 0.82,
        "critique": "Good",
        "skipped": False,
        "skip_reason": None,
        "judge_model": "qwen/qwen-max",
    }
    score_obj.skipped = False
    score_obj.faithfulness = 0.9
    score_obj.relevance = 0.8
    score_obj.completeness = 0.7
    score_obj.overall = 0.82

    with patch("api.routes.evaluate.QualityEvaluator.score", AsyncMock(return_value=score_obj)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/evaluate",
                json={
                    "items": [
                        {
                            "item_id": "test-1",
                            "question": "What does verify_password do?",
                            "answer": "It verifies plaintext against bcrypt hash.",
                            "context_chunks": ["def verify_password(...): return bcrypt.checkpw(...)"],
                        }
                    ]
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 1
    assert body["aggregate"]["avg_overall"] > 0


@pytest.mark.asyncio
async def test_post_evaluate_rejects_empty_items(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/evaluate", json={"items": []})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_post_evaluate_rejects_more_than_50_items(app: FastAPI):
    items = [
        {"question": "q", "answer": "a"}
        for _ in range(51)
    ]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/evaluate", json={"items": items})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_metrics_returns_sections(app: FastAPI):
    summary = DailyCostSummary(
        date="2025-01-01",
        total_cost_usd=0.12,
        total_tokens=1200,
        budget_limit_usd=5.0,
        budget_used_pct=2.4,
        remaining_usd=4.88,
        over_budget=False,
        per_provider={
            "openrouter": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
                "call_count": 1,
                "cost_today_usd": 0.12,
            }
        },
    )

    fake_cache = MagicMock()
    fake_cache.get_stats.return_value = {
        "embedding": {"hits": 1, "misses": 1, "total": 2, "hit_rate": 0.5, "key_count": 1},
        "search": {"hits": 1, "misses": 1, "total": 2, "hit_rate": 0.5, "key_count": 1},
        "graph": {"hits": 1, "misses": 1, "total": 2, "hit_rate": 0.5, "key_count": 1},
        "redis_memory_bytes": 1024,
    }

    fake_semantic = MagicMock()
    fake_semantic.get_stats = AsyncMock(return_value={"collection_name": "semantic_answer_cache", "status": "green", "total_points": 3})

    fake_redis = MagicMock()
    fake_redis.get.side_effect = lambda key: {
        "eval:avg:faithfulness": "0.9",
        "eval:avg:relevance": "0.8",
        "eval:avg:completeness": "0.7",
        "eval:avg:overall": "0.82",
        "eval:count": "2",
    }.get(key)

    statuses = {
        "openrouter": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
        "gemini": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
        "deepseek": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
    }

    with patch("api.routes.metrics._tracker.get_daily_summary", AsyncMock(return_value=summary)):
        with patch("api.routes.metrics.get_cache_manager", return_value=fake_cache):
            with patch("api.routes.metrics.get_semantic_answer_cache", return_value=fake_semantic):
                with patch("api.routes.metrics.redis.Redis.from_url", return_value=fake_redis):
                    with patch("api.routes.metrics.CircuitBreaker.get_all_statuses", return_value=statuses):
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get("/api/v1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "token_usage" in body
    assert "budget" in body
    assert "cache" in body
    assert "circuit_breakers" in body
    assert "eval_scores" in body


@pytest.mark.asyncio
async def test_get_metrics_export_returns_attachment_header(app: FastAPI):
    summary = DailyCostSummary(
        date="2025-01-01",
        total_cost_usd=0.0,
        total_tokens=0,
        budget_limit_usd=5.0,
        budget_used_pct=0.0,
        remaining_usd=5.0,
        over_budget=False,
        per_provider={"openrouter": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "call_count": 0, "cost_today_usd": 0.0}},
    )

    fake_cache = MagicMock()
    fake_cache.get_stats.return_value = {"embedding": {}, "search": {}, "graph": {}, "redis_memory_bytes": 0}

    fake_semantic = MagicMock()
    fake_semantic.get_stats = AsyncMock(return_value={"collection_name": "semantic_answer_cache", "status": "green", "total_points": 0})

    statuses = {
        "openrouter": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
        "gemini": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
        "deepseek": SimpleNamespace(state=CircuitState.CLOSED, failure_count=0, last_failure_time=0.0, last_success_time=0.0),
    }

    with patch("api.routes.metrics._tracker.get_daily_summary", AsyncMock(return_value=summary)):
        with patch("api.routes.metrics.get_cache_manager", return_value=fake_cache):
            with patch("api.routes.metrics.get_semantic_answer_cache", return_value=fake_semantic):
                with patch("api.routes.metrics.CircuitBreaker.get_all_statuses", return_value=statuses):
                    with patch("api.routes.metrics.redis.Redis.from_url", side_effect=Exception("no redis")):
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.get("/api/v1/metrics/export")

    assert response.status_code == 200
    assert "Content-Disposition" in response.headers
    assert "filename=" in response.headers["Content-Disposition"]


@pytest.mark.asyncio
async def test_post_webhook_push_returns_202_and_task(app: FastAPI):
    fake_db = AsyncMock()
    fake_db.execute = AsyncMock(return_value=_ScalarResult(None))
    fake_db.add = MagicMock()
    fake_db.flush = AsyncMock()
    fake_db.commit = AsyncMock()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db

    with patch("api.routes.webhook.run_ingestion_task.delay", return_value=SimpleNamespace(id="task-123")):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ingest/webhook",
                headers={"X-GitHub-Event": "push", "Content-Type": "application/json"},
                json={
                    "ref": "refs/heads/main",
                    "repository": {
                        "clone_url": "https://github.com/tiangolo/fastapi.git",
                        "html_url": "https://github.com/tiangolo/fastapi",
                    },
                },
            )

    assert response.status_code == 202
    body = response.json()
    assert body["triggered"] is True
    assert body["task_id"] == "task-123"
    assert body["branch"] == "main"


@pytest.mark.asyncio
async def test_post_webhook_ping_ignored(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingest/webhook",
            headers={"X-GitHub-Event": "ping", "Content-Type": "application/json"},
            json={"zen": "keep it logically awesome"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["triggered"] is False


@pytest.mark.asyncio
async def test_post_webhook_invalid_json_returns_400(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ingest/webhook",
            headers={"X-GitHub-Event": "push", "Content-Type": "application/json"},
            content="{not-json",
        )

    assert response.status_code == 400
